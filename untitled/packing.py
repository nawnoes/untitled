from typing import Dict, Optional, List, Union
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def pack_dataset(dataset, key2length, keys):
    shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
    if keys is None:
        keys = list(shapes.keys())
    for k in keys:
        if k not in shapes:
            raise ValueError(f'Key {k} not found in dataset, Available keys are {shapes.keys()}')
        if not shapes[k].is_compatible_with(tf.TensorShape([None])):
            raise ValueError(f'Tensors to be packed must be one-dimensional')
        
    if isinstance(key2length, int):
        key2length = {k: key2length for k in keys}
    
    for k in keys:
        for suffix in ['_segmentation', '_position']:
            key2length[k + suffix] = key2length[k]
            
    dataset = dataset.map(
        lambda x: {k: x[k][:key2length[k]] for k in keys},
        num_parallel_calls=AUTOTUNE
    )
    
    batch_size = max(key2length.values())
    dataset = dataset.padded_batch(batch_size, padded_shape={k:[-1] for k in keys})
    dataset = _pack_with_tf_ops(dataset, keys, key2length)
    
    def my_fn(x):
        return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}
    
    return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)
        
def _pack_with_tf_ops(dataset, keys, key2length):
    empty_example = {}
    for k in keys:
        empty_example[k] = tf.zeros([0], dtype=tf.int32)
        empty_example[k+'_position'] = tf.zeros([0], dtype=tf.int32)
    
    keys_etc = empty_example.keys()
    
    def write_packed_example(partial, outputs):
        new_partial = empty_example.copy()
        new_outputs = {}
        
        for k in keys_etc:
            new_outputs[k] = outputs[k].write(
                outputs[k].size(), 
                tf.pad(partial[k],[[0, key2length[k] - tf.size(partial[k])]])
            )

        return new_partial, new_outputs
    
    def map_fn(x):
        partial = empty_example.copy()
        i = tf.zeros([], dtype=tf.int32)
        dynamic_batch_size = tf.shape(x[keys[0]])[0]
        outputs = {}
        
        for k in keys:
            outputs[k] = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
            outputs[k+'_position'] = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
        
        def body_fn(i, partial, outputs):
            can_append = True
            one_example = {}
            
            for k in keys:
                val = tf.cast(x[k][i], tf.int32)
                val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
                one_example[k] = val
                
            for k in keys:
                can_append = tf.logical_and(
                    can_append, 
                    tf.less_equal(tf.size(partial[k]) + tf.size(one_example[k]), key2length[k])
                )
            
            def false_fn():
                return write_packed_example(partial, outputs)
            
            def true_fn():
                return partial, outputs
            
            partial, outputs = tf.cond(can_append, true_fn, false_fn)
            
            new_partial = {}
            for k in keys:
                new_seq = one_example[k][:key2length[k]]
                new_seq_len = tf.size(new_seq)
                
                new_partial[k] = tf.concat([partial[k], new_seq], 0)
                new_partial[k + '_position'] = tf.concat(partial[k+'_position', tf.range(new_seq_len)], 0)
            
            partial = new_partial
            return i + 1, partial, outputs
        
        i, partial, outputs = tf.while_loop(
            cond=lambda *_: True,
            body=body_fn,
            loop_vars=(i, partial, outputs),
            shape_invariants=(
                tf.TensorShape([]),
                {k: tf.TensorShape([None]) for k in keys_etc},
                {k: tf.TensorShape(None) for k in keys_etc}
            ),
            maximum_iterations=dynamic_batch_size
        )
        
        _, outputs = write_packed_example(partial, outputs)
        packed = {k: outputs[k].stack() for k in keys_etc}
        for k in keys:
            packed[f'{k}_segmentation'] = (
                tf.cumsum(tf.cast(tf.equal(packed[f'{k}_position'], 0), tf.int32), axis=1) *
                tf.cast(tf.not_equal(packed[k], 0), tf.int32)
            )
            
        return packed
    dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
    return dataset