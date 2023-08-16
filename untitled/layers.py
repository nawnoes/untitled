import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import numpy as np

import jax
from jax import lax
from jax import random
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

withLP = nn.with_logical_partitioning
ScanIn = nn_partitioning.ScanIn

Config = Any

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]

# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)

def dot_product_attention(query, 
                          key, 
                          value, 
                          bias: Optional[Array] = None, 
                          dropout_rng: Optional[PRNGKey] = None, 
                          dropout_rate: float = 0., 
                          deterministic: bool = False, 
                          dtype: jnp.dtype = jnp.float32, 
                          float32_logits: bool = False):
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
    
    query = LayerNorm(dtype=dtype, name='query_layer_norm', kernel_axes = ('head',))(query)
    key = LayerNorm(dtype=dtype, name='key_layer_norm', kernel_axes = ('head',))(key)
    
    attn_weight = jnp.einsum('bqhd,bkhd->bhqk', query, key)
    
    if bias is not None:
        attn_weight = attn_weight + bias.astype(attn_weight.dtype)
    
    attn_weight = jax.nn.softmax(attn_weight).astype(dtype)
    
    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = list(attn_weight.shape)
        dropout_shape[-2] = 1 # bhqk -> bh1k
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weight.shape)
        
        multiplier = keep.astype(attn_weight.dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weight = attn_weight * multiplier
        
    return jnp.einsum('bhqk,bkhd->bqhd', attn_weight, value)

def nd_dense_init(scale, mode, distribution):
    "Initializer with in_axis, out_axis set at call time."
    def init_fn(key, shape, dtype, in_axis, out_axis):
        fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
        return fn
    return init_fn

def _normalize_axes(axes, ndim):
    return tuple(ax if ax>=0 else ndim + ax for ax in axes)

def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)

class DenseGeneral(nn.Module):
    pass

def _convert_to_activationf_function(fn_or_string):
    if fn_or_string == 'linear':
        return lambda x: x
    elif isinstance(fn_or_string, str):
        return getattr(nn. fn_or_string)
    elif callable(fn_or_string):
        return fn_or_string

class MultiHeadDotProductAttention(nn.Module):
    pass

class MLPBlock(nn.Module):
    pass

class LayerNorm(nn.Module):
    pass

class Embedding(nn.Module):
    pass

class RelativePositionBiases(nn.Module):
    pass

def make_attention_mask():
    pass

def make_causal_mask():
    pass

def combine_masks():
    pass

def combine_biases():
    pass

def make_decoder_mask():
    pass

class DecoderLayer(nn.Module):
    pass

class Decoder(nn.Module):
    pass

class DecoderOnlyTransformer(nn.Module):
    pass