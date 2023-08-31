import os
from typing import Optional
import functools

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from jax.sharding import PartitionSpec as P

from utils import log

import tokenizer
import multihost_dataloader
import sequence_packing

AUTOTUNE = tf.data.experimental.AUTOTUNE

def shift_right_tf(x, axis=1):
    """Shift the input to right by padding and slicing on axis"""
    pad_widths = [(0,0)] * len(x.shape)
    pad_widths[axis] = (1,0)
    
    slices = [slice(None),] * len(x.shape)
    slices[axis] = slice(0, -1)
    
    padded = tf.pad(
        x,
        tf.constant(pad_widths),
        mode='CONSTANT',
        constant_values=tf.constant(0, x.dtype)
    )
    
    return padded[tuple(slices)]

def shift_inputs_tf(x, segment_ids=None, axis=1):
    """Shift inputs and EOS by 0 for packed inputs"""
    shifted = shift_right_tf(x, axis=axis)
    
    if segment_ids is not None:
        shifted *= tf.cast(segment_ids == shift_right_tf(segment_ids, axis=axis), x.dtype)
    
    return shifted

def shift_data(x, axis=0, segmented=True):
    segment_ids = x['inputs_segmentation'] if segmented else None
    x['inputs'] = shift_inputs_tf(x['inputs', segment_ids=segment_ids, axis=axis])
    return x

def normalize_features(ds):
    def _normalize_features(features):
        features['inputs'] = features.pop('text')
        features['targets'] = features['inputs']
        return features
    
    return ds.map(_normalize_features, num_parallel_calls=AUTOTUNE)

def preprocessing_pipeline(dataset,
                           batch_size,
                           global_mesh,
                           shuffle,
                           num_epochs=1,
                           pack_examples=True,
                           shuffle_buffer_size=1024,
                           max_length=512,
                           shift=True,
                           drop_remainder=True,
                           prefetch_size=tf.data.experimental.AUTOTUNE,
                           data_sharding=None,
                           data_shuffle_seed=0):
    """Shuffle and batch/pack the given dataset"""
    
    # Max length filter
    def length_filter(max_len):
        def filter_fn(x):
            source, target = x['inputs'], x['targets']
            l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
            return tf.less(l, max_len +1)
        return filter_fn
    
    if max_length > 0:
        datasets = dataset.filter(length_filter(max_length))
        
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed = data_shuffle_seed)
    
    # Repeat
    dataset = dataset.repeat(num_epochs)
    
    # Packing
    if pack_examples:
        dataset = sequence_packing.pack_dataset(dataset, max_length)
    
    # Shift inputs for teacher forcing
    if shift:
        dataset = dataset.map(
            functools.partial(shift_data, axis=0, segmented=pack_examples),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )
        
    # Multihost dataloading: sharding and jax.Array prep function
    dataset_structure = tf.data.experimental.get_structure(dataset)
    global_data_shape = jax.tree_map(lambda x: P(*data_sharding), dataset_structure)
    data_axes = jax.tree_map(lambda x: P(*data_sharding), dataset_structure)

def get_datasets():
    pass

def preprocess_dataset():
    pass