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
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]

# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, Dtype, InitializerAxis, InitializerAxis], Array]

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
    features: Union[Iterable[int], int]
    config: Config
    axis: Union[Iterable[int], int] = -1
    dtype: Dtype = jnp.float32
    kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal') # distribution scale 1.0, 'fan_in' use for standard deviation sqrt(scale/n).
    kernel_axes: Tuple[str, ...] = ()
    
    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        config = self.config
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)
        
        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(self.axis, inputs.ndim)
        
        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel_in_axis = np.arange(len(axis))
        kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
        
        kernel = self.param(
            'kernel',
            nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
            kernel_shape,
            jnp.float32,
            kernel_in_axis,
            kernel_out_axis
        )
        kernel = jnp.asarray(kernel, self.dtype)
        
        contract_ind = tuple(range(0, len(axis)))
        return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((),())))

def _convert_to_activationf_function(fn_or_string):
    if fn_or_string == 'linear':
        return lambda x: x
    elif isinstance(fn_or_string, str):
        return getattr(nn. fn_or_string)
    elif callable(fn_or_string):
        return fn_or_string

class MultiHeadDotProductAttention(nn.Module):
    num_heads: int
    head_dim: int
    config: Config
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.
    kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal')
    float32_logits: bool = False
    
    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 bias: Optional[Array] = None,
                 *,
                 decode: bool = False,
                 deterministic: bool = False) -> Array:
        config = self.config
        projection = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.num_heads, self.head_dim),
            kernel_axes=('embed', 'heads', 'kv'),
            dtype=self.dtype,
            config=config
        )
        # rescale the attention logits by 1/sqrt(depth_kq)
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
        def query_init(*args):
            return self.kernel_init(*args) / depth_scaling
        
        # project input to multi-headed q/k/v
        query = projection(kernel_init=query_init, name='query')(inputs_q)
        key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
        value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)
        
        query = nn.with_logical_partitioning(query, ('activation_bathc', 'activation_length', 'activation_heads', 'activation_kv'))
        query = checkpoint_name(query, 'query_proj')
        
        key = nn.with_logical_partitioning(key, ('activation_bathc', 'activation_length', 'activation_heads', 'activation_kv'))
        key = checkpoint_name(key, 'key_proj')
        
        value = nn.with_logical_partitioning(value, ('activation_bathc', 'activation_length', 'activation_heads', 'activation_kv'))
        value = checkpoint_name(value, 'value_proj')
        
        if decode:
            # Need to add cache for fast decoding
            pass
        
        if mask is not None:
            attention_bias = lax.select(
                mask>0,
                jnp.full(mask.shape, 0.).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype)
            )
        else:
            attention_bias = None
        
        if bias is not None: # add provided bias term(e.g. relative position embedding)
            attention_bias = combine_biases(attention_bias, bias)
            
        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng('dropout')
            
        x = dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            float32_logits=self.float32_logits
        )
        
        out = DenseGeneral(
            feature=inputs_q.shape[-1],
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            kernel_axes=('head', 'kv', 'embed'),
            dtype=self.dtype,
            name='out',
            config=config
        )(x)
        
        return out
        

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