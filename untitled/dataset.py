import os
from typing import Optional
import functools

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from jax.sharding import PartitionSpec as P

from log import log

import tokenizer
import multihost_dataloader
import packing

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
    x['inputs'] = shift_inputs_tf(x['inputs'], segment_ids=segment_ids, axis=axis)
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
        dataset = packing.pack_dataset(dataset, max_length)
    
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
    
    assert batch_size % global_mesh.size == 0, 'Batch size should be divisible number of global devices'
    
    # Batch exampels
    if pack_examples:
        dataset = dataset.batch(batch_size // jax.process_count(), drop_remainder=drop_remainder)
    else:
        dataset = dataset.padded_batch(
            batch_size // jax.process_count(),
            padded_shapes={'inputs': max_length, 'targets': max_length},
            padding_values={'inputs': 0, 'targets': 0}, # 수정 필요할 수도
            drop_remainder=drop_remainder
        )
    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)
    
    multihost_gen = (
        multihost_dataloader.get_batch_sharded_data_pipeline(
            dataset, data_sharding, global_data_shape, global_mesh, data_axes
        )
    )
    
    return multihost_gen

def get_datasets(config, read_config):
    """Load and retrun dataset of batched examples for use during training"""
    train_ds_builder = tfds.builder(config.dataset_name)
    train_ds = train_ds_builder.as_dataset(
        split='train',
        read_config=read_config,
        shuffle_files=config.enable_data_shuffling
    )
    train_ds = train_ds.shard(num_shards=jax.process_count(), index=jax.process_index())
    train_ds = normalize_features(train_ds)
    
    if config.eval_dataset_name:
        eval_ds_builder = tfds.builder(config.eval_dataset_name)
    else:
        eval_ds_builder = train_ds_builder
        
    eval_ds = eval_ds_builder.as_dataset(
        split=config.eval_spilt,
        read_config=read_config,
        shuffle_files=config.enable_data_shuffling
    )
    eval_ds = eval_ds.shard(num_shards=jax.process_count(), index=jax.process_index())
    eval_ds = normalize_features(eval_ds)
    
    return train_ds, eval_ds


def preprocess_dataset(config, global_mesh, train_ds, eval_ds, vocab_path, data_shuffle_seed=0):
    sp_tokenizer = tokenizer.load_sentencepiece_tokenizer(vocab_path)
    
    train_ds = train_ds.map(tokenizer.TokenizerOp(sp_tokenizer), num_parallel_call=AUTOTUNE)
    eval_ds = eval_ds.map(tokenizer.TokenizerOp(sp_tokenizer), num_parallel_call=AUTOTUNE)
    
    # set global batch size
    batch_size = config.per_device_batch_size * global_mesh.size
    if config.eval_per_device_batch_size > 0:
        eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
    else:
        eval_batch_size = batch_size
    def filter_keys(record):
        return {'inputs':record['inputs'], 'targets':record['targets']}
    
    train_ds = train_ds.map(filter_keys, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_ds.map(filter_keys, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_iter = preprocessing_pipeline(
        train_ds,
        batch_size,
        global_mesh,
        shuffle=config.enable_data_shuffling,
        num_epochs=None,
        pack_examples=True,
        max_length=config.max_target_length,
        shift=True,
        data_sharding=config.data_sharding,
        data_shuffle_seed=data_shuffle_seed
    )
    
    eval_iter = preprocessing_pipeline(
        eval_ds,
        eval_batch_size,
        global_mesh,
        shuffle=config.enable_data_shuffling,
        pack_examples=False,
        max_length=config.max_eval_target_length,
        shift=False,
        data_sharding=config.data_sharding,
        data_shuffle_seed=data_shuffle_seed
    )
    
    predict_iter = preprocessing_pipeline(
        eval_ds,
        eval_batch_size,
        global_mesh,
        shuffle=config.enable_data_shuffling,
        pack_examples=False,
        max_length=config.max_predict_length,
        shift=False,
        drop_remainder=False,
        data_sharding=config.data_sharding,
        data_shuffle_seed=data_shuffle_seed
    )
    
    return train_iter, eval_iter, predict_iter, sp_tokenizer