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

def dot_product_attention(query, key, value, bias, dropout_rng, dropout_rate, deterministic, dtype, float32_logits):
    pass

def nd_dense_init(scale, mode, distribution):
    pass

def _normalize_axes(axes, ndim):
    pass

def _canonicalize_tuple(x):
    pass

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