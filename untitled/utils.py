import checkpointing
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils

from jax.experimental.pjit import pjit

import json
import flax
from flax.training import train_state
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import optax
import os
import subprocess
from log import log
    
def l2norm_pytree(x):
    return jax.tree_util.tree_reduce(
        lambda x, y: x + jax.numpy.sum(y ** 2), x, initializer=0.0
    ) ** 0.5

def activate_profiler(config):
    if config.enable_profiler:
        jax.profiler.start_trace(config.tensorboard_dir)

def deactivate_profiler(config):
    if config.enable_profiler:
        jax.profiler.stop_trace()

def _prepare_metrics_for_json(metrics, step, run_name):
    metrics_dict = {}
    for val in metrics['scalar']:
        metrics_dict[val] = float(metrics['scalar'][val])
    
    metrics_dict['step'] = float(step)
    metrics_dict['run_name'] = run_name
    return metrics_dict

def write_metrics_locally(metrics, step, config, file):
    if step == 0:
        file.truncate(0)
    
    metrics_dict = _prepare_metrics_for_json(metrics, step, config.run_name)
    file.write(str(json.dumps(metrics_dict)) + '\n')

def write_metrics_for_gcs(metrics, step, config, running_metrics):
    metrics_dict_step = _prepare_metrics_for_json(metrics, step, config.run_name)
    running_metrics.append(metrics_dict_step)
    
    if (step + 1) % config.log_period == 0 or step == config.steps - 1:
        start_step = (step // config.log_period) * config.log_period
        metrics_filename = f'metrics_step_{start_step:06}_to_step_{step:06}.txt'
        with open(metrics_filename, 'w', encoding='utf8') as metrics_for_gcs:
            for metrics_step in running_metrics:
                metrics_for_gcs.write(str(json.dumps(metrics_step)) + '\n')
        
        metrics_for_gcs.close()
        gcs_filename = os.path.join(config.metrics_dir, metrics_filename)
        command = ['gsutil', 'mv', metrics_filename, gcs_filename]
        
        log(f'Moving file {metrics_filename} to GCS...')
        subprocess.run(command, check=True, capture_output=True)
        log(f'File {metrics_filename} moved successfully')
        running_metrics = []
        
    return running_metrics

def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
    """Evaluates unspecified DCN/ICI parallelism values"""
    if -1 in parallelism_vals:
        assert (parallelism_vals.count(-1) == 1, 
                f'Found unspecified values (-1) for more than one {parallelism_type} parallelism aixs. At most one axis can be unspecified')
        
        determined_val = target_product / np.product(parallelism_vals) * -1
        
        assert determined_val >= 1 and determined_val.is_integer, f"Unspecified value unable to be determined with the given\
            {parallelism_type} parallelism values"
            
        parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)
    
    target_type = 'slices' if parallelism_type == 'DCN' else 'devices per slice'
    
    assert np.product(parallelism_vals) == target_product, f' Number of {target_type} {target_product} does not match\
        the product of the {parallelism_type} parallelism {np.product(parallelism_vals)}'
        
    return parallelism_vals

def create_device_mesh(config, logging):
    """Creates a device mesh with each slice in its own data parallel group. if there is only one slice, use two replicas"""
    devices = jax.devices()
    num_devices = len(devices)
    
    try:
        num_slices = 1 + max([d.slice_index for d in devices])
    except:
        num_slices = 1
        
    num_devices_per_slice = num_devices // num_slices
    log(f'Devices: {devices} (num_deivces: {num_devices})')
    assert len(devices) > 1, 'You must have at least two devices'
    
    multi_slice_env = hasattr(jax.devices()[0], 'slice_index')
    
    dcn_parallelism = [config.dcn_data_parallelism, config.dcn_fsdp_parallelism, config.dcn_tensor_parallelism]
    ici_parallelism = [config.ici_data_parallelism, config.ici_fsdp_parallelism, config.ici_tensor_parallelism]
    
    dcn_parallelism = fill_unspecified_mesh_axes(dcn_parallelism, num_slices, 'DCN')
    ici_parallelism = fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, 'ICI')
    
    if multi_slice_env: # 여러개의 slice를 사용하는 경우 DCN parallelism 까지 같이 사용
        mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
    else:
        mesh = mesh_utils.create_device_mesh(ici_parallelism) # device에 대한 mesh 생성.
    
    log(f'Decided on mesh: {mesh}')
    
    return mesh
        

def unbox_logicaly_partitioned_trainstate(boxed_train_state):
    """Unboxes the flax.LogicallyPratitioned pieces in a train state"""
    return jax.tree_util.tree_map(
        lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
        boxed_train_state,
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned)
    )

def init_train_state(model, optimizer, config, key):
    input_shape = (len(jax.devices()) * config.per_device_batch_size, config.max_target_length)
    model_variables = model.init(
        {'params':key, 'droupout': key},
        jnp.ones(input_shape),
        jnp.ones(input_shape)
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model_variables['params'],
        tx=optimizer
    )
    return state

def setup_initial_state(model, optimizer, config, rng, mesh, checkpoint_manager):
    init_train_state_partial = functools.partial(init_train_state, model, optimizer, config)
    abstract_state = jax.eval_shape(init_train_state_partial, rng)
    state_logical_annotations = nn.get_partition_spec(abstract_state)
    unboxed_abstract_state = unbox_logicaly_partitioned_trainstate(abstract_state)
    
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
        state, raw_params = checkpointing.load_state_if_possible(
            checkpoint_manager,
            config.load_parameters_path,
            config.load_from_other_directory,
            config.load_from_other_directory_step,
            unboxed_abstract_state,
            mesh,
            state_mesh_annotations
        )
        if not state:
            state = pjit(
                init_train_state_partial,
                in_shardings=None,
                out_shardings=state_mesh_annotations
            )(rng)
            if raw_params:
                state = state.replace(params = raw_params)
        raw_params = None
    
    state = unbox_logicaly_partitioned_trainstate(state)
    return state, state_mesh_annotations

def rsqrt_schedule(init_value, shift):
    def schedule(count):
        return init_value * (1 + count + shift) ** -0.5 * shift ** 0.5
    return schedule

def create_learning_rate_schedule(learning_rate, warmup_steps):
    return optax.join_schedules([
        optax.linear_schedule(
            init_value=0,
            end_value=learning_rate,
            transition_steps=warmup_steps
        ),
        rsqrt_schedule(init_value=learning_rate, shift=warmup_steps)
    ], boundaries=[warmup_steps])