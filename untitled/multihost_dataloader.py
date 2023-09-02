from collections import defaultdict  # pylint: disable=g-importing-member
from dataclasses import dataclass  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
import os
from typing import Callable, Any, Dict, List, Tuple, Optional
import tensorflow as tf  # pylint: disable=g-import-not-at-top
import time
import numpy as np

import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh

from utils import log


Pytree = Any
Device = Any

DATA_DIM = 0

def check_inputs(dataset, global_data_shape, data_axes):
    dataset_structure = jax.tree_util.tree_structure(
        tf.data.experimenta.get_structure(dataset)
    )
    global_data_shape_structure = jax.tree_util.tree_structure(global_data_shape)
    data_axes_structure = jax.tree_util.tree_structure(data_axes)
    
    try:
        assert(dataset_structure == global_data_shape_structure == data_axes_structure), 'All inputs should have the same pytree structure'
    except AssertionError as msg:
        log(f'{msg} - global shapes should be array or classes not tuples, otherwise tree map enumerate individual dimensions as leaves')
        log(f'Dataset: {dataset_structure}')
        log(f'Shape  : {global_data_shape_structure}')
        log(f'Axes   : {data_axes_structure}')
        
    shapes, _ = jax.tree_util.tree_flatten(global_data_shape)
    batch_dims = [s[0] for s in shapes]
    assert all(b == batch_dims[0] for b in batch_dims), 'All batch axis should be equal for gdas'
    assert all(b[0] == shapes[0][0] for b in shapes), 'All dataset elements should be sharded along the data axis indentically'
    
    batch_dim = batch_dims[0]
    return batch_dim

def get_batch_sharded_data_pipeline(dataset, data_sharding, global_data_shape, global_mesh, data_axes):
    """Each device loads batch_size/num_devices
    Each host first loads batch_size/num_hosts then, shards that eqaully across it's devices

    Args:
        dataset: tf dataset over all files
        data_sharding: data sharding axes
        global_data_shape: what the size of the GDA should be
        global_mesh: global devices mesh
        data_axes: axes along which data is partitioned
    Returns:
        sharded_dataset: per host dataset
    """
    _ = check_inputs(dataset, global_data_shape, data_axes)
    
    dataset = iter(dataset.as_numpy_iterator())
    
    multihost_generator = partial(
        get_next_batch_sharded,
        dataset,
        data_sharding,
        global_data_shape,
        global_mesh
    )
    return multihost_generator

def get_next_batch_sharded(local_dataset, data_sharding, global_data_shape, global_mesh):
    """Split the host loaded data equally over all devices"""
    SLEEP_TIME = 10
    MAX_DATA_LOAD_ATTEMPTS = 30
    
    data_load_attempts = 0
    loaded_data_sucees = False
    while not loaded_data_sucees and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
        data_load_attempts += 1
        try:
            local_data = local_dataset.next()
            loaded_data_sucees = True
        except tf.errors.FailedPreconditionError:
            log("Faild to get next data batch")
            time.sleep(SLEEP_TIME)
        
    local_devices = global_mesh.local_devices
    local_device_count = jax.local_device_count()
    
    def _put_to_devices(x):
        try:
            per_device_arrays = np.split(x, local_device_count, axis=0)
        except ValueError as e:
            raise ValueError(
                f'Unable to put to devices shape {x.shape} with local device count {local_device_count}'
            ) from e
        
        device_buffers = [
            jax.device_put(arr, d) # Transfers x to device
            for arr, d in zip(per_device_arrays, local_devices)
        ]
        return device_buffers
    
    input_sharding_constraint = PartitionSpec(*data_sharding, None)
    
    def form_gda(local_data, shape):
        device_buffers = _put_to_devices(local_data)
        shape = tuple(shape)
        input_gda = jax.make_array_from_single_device_arrays(
            shape,
            jax.sharding.NamedSharding(
                global_mesh,
                input_sharding_constraint
            ),
            device_buffers
        )
        return input_gda
    
    input_gdas = jax.tree_map(form_gda, local_data, global_data_shape)
    
    return input_gdas