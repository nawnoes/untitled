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
        
        query = nn.with_logical_constraint(query, ('activation_bathc', 'activation_length', 'activation_heads', 'activation_kv'))
        query = checkpoint_name(query, 'query_proj')
        
        key = nn.with_logical_constraint(key, ('activation_bathc', 'activation_length', 'activation_heads', 'activation_kv'))
        key = checkpoint_name(key, 'key_proj')
        
        value = nn.with_logical_constraint(value, ('activation_bathc', 'activation_length', 'activation_heads', 'activation_kv'))
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
    config: Config
    intermediate_dim: int = 2048
    activations: Sequence[Union[str, Callable]] = ('relu',)
    kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        config = self.config
        activations = []
        for idx, act_fn in enumerate(self.activations):
            dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
            
            x = DenseGeneral(
                self.intermediate_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                kernel_axes=('embed', 'mlp'),
                name=dense_name,
                config=config
            )(inputs)
            
            x = _convert_to_activationf_function(act_fn)(x)
            activations.append(x)
        
        x = functools.reduce(operator.mul, activations)
        x = nn.Dropout(
            rate=self.intermediate_dropout_rate,
            broadcast_dims=(-2,)
        )(x, deterministic=deterministic)
        
        x = nn.with_logical_constraint(x, ('activation_batch', 'activation_length', 'activation_mlp'))
        output = DenseGeneral(
            inputs.shape[-1],
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=('mlp', 'embed'),
            name='wo',
            config=config
        )(x)
        
        return output

class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    kernel_axes: Tuple[str, ...] = ()
    scale_init: Initializer = nn.initializers.ones
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]
        
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
        
        scale = self.param(
            'scale',
            nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
            (features,),
            jnp.float32
        )
        scale = jnp.asarray(scale, self.dtype)
        
        return y * scale

class Embedding(nn.Module):
    num_embeddings: int
    features: int
    cast_input_dtype: Optional[Dtype] = None
    dtype: Dtype = jnp.float32
    attend_dtype: Optional[Dtype] = None
    embedding_init: Initializer = default_embed_init
    embedding: Array = dataclasses.field(init=False)
    
    def setup(self):
        self.embedding = self.param(
            'embedding',
            nn.with_logical_partitioning(self.embedding_init, ('vocab', 'embed')),
            (self.num_embeddings, self.features),
            jnp.float32
        )
    
    def __call__(self, inputs: Array) -> Array:
        if self.cast_input_dtype:
            inputs = inputs.astype(self.cast_input_dtype)
        output = jnp.asarray(self.embedding, self.dtype)[inputs]
        output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'activation_embed'))
        return output

class RelativePositionBiases(nn.Module):
    num_buckets: int
    max_distance: int
    num_heads: int
    dtype: Any
    embedding_init: Callable[..., Array] = nn.linear.default_embed_init
    
    @staticmethod
    def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_bucket=32,
                                max_distance=128):
        """Translate relative position to a bucket number for relative attention
        
        the relative position is defined as memory_position - query_position
        i.e. the distance in token from the attending position to the attended to position.
        If bidirectional=False then positive relative position are invalid
        
        we use smaller buckets for small absolute relative postion and larger buckeys for larger
        absolute relative positions. All relative positions >= max distance map to the same bucket.
        All relative positions <= max_distance map to the same bucket. 
        
        This should allow for more graceful generalization to longer sequences than the model 
        has been trained on.
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_bucket //= 2
            ret += (n < 0).astype(np.int32) * num_bucket
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)
            
        max_exact = num_bucket // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
                                    np.log(max_distance / max_exact) * (num_bucket - max_exact)).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_bucket - 1)
        ret += np.where(is_small, n, val_if_large)
        
        return ret
    @nn.compact
    def __call__(self, q_len, k_len, bidirectional=True):
        context_position = np.arange(q_len, dtype=jnp.int32)[:, None]
        memory_position = np.arange(k_len, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position
        
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_bucket=self.num_buckets,
            max_distance=self.max_distance
        )
        relative_attention_bias = self.param(
            'rel_embedding',
            nn.with_logical_partitioning(self.embedding_init, ('heads', 'relpos_buckets')),
            (self.num_heads, self.num_buckets),
            jnp.float32
        )
        relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
        
        values = lax.dot_general(
            relative_attention_bias, 
            rp_bucket_one_hot,
            (((1,), (0,)),
            ((), ()))
        )
        return values[jnp.newaxis, ...]

def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable =jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: Dtype = jnp.float32):
    mask = pairwise_fn(
        jnp.expand_dims(query_input, axis=-1), # [batch, len_q] -> [batch, len_q, 1]
        jnp.expand_dims(key_input, axis=-2) # [batch, len_q] -> [batch, 1, len_kv]
    )
    
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, asix=tuple(range(extra_batch_dims)))
    
    return mask.astype(dtype)
    

def make_causal_mask(x:Array,
                     extra_batch_dims: int = 0,
                     dtype: Dtype = jnp.float32):
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        jnp.greater_equal,
        extra_batch_dims=extra_batch_dims,
        dtype=dtype
    )

def combine_masks(*masks: Optional[Array],
                  dtype: Dtype = jnp.float32):
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}'
    
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)

def combine_biases():
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask

def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: Dtype,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None):
    masks = []
    causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)
    
    # Positions with value 1 in 'decoder_causal_attention' can attend bidirectionally.
    if decoder_causal_attention is not None:
        # [batch, 1, length, length]
        inputs_mask = make_attention_mask(
            decoder_causal_attention,
            decoder_causal_attention,
            jnp.logical_and,
            dtype=dtype
        )
        masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
    else:
        masks.append(causal_mask)
        
    # Padding mask

class DecoderLayer(nn.Module):
    config: Config
    
    @nn.compact
    def __call__(self,
                 inputs,
                 decoder_mask,
                 deterministic,
                 decode,
                 max_decode_length):
        config = self.config
        
        
        input_length = inputs.shape[-2]
        decoder_bias = RelativePositionBiases(
            num_buckets=32,
            max_distance=128,
            num_heads=config.num_heads,
            dtype=config.dtype,
            embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
            name='relpos_bias'
        )(input_length, input_length, False)
        
        inputs = nn.with_logical_constraint(inputs, ('activation_batch', 'activation_length', 'activation_embed'))
        
        layer_norm_output = LayerNorm(
            dtype=config.dtype,
            name='pre_self_attention_layer_norm',
            kernel_axes=('embed',)
        )(inputs)
        
        layer_norm_output = nn.with_logical_constraint(layer_norm_output, ('activation_batch', 'activation_length', 'activation_embed'))
        
        attention_output = MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            head_dim=config.head_dim,
            dropout_rate=config.dropout_rate,
            name='self_attention',
            config=config
        )(layer_norm_output, layer_norm_output, decoder_mask, decoder_bias, deterministic=deterministic, decode=decode)
        
        attention_output = nn.with_logical_constraint(attention_output, ('activation_batch', 'activation_length', 'activation_embed'))
        
        mlp_output = MLPBlock(
            intermediate_dim=config.mlp_dim,
            activations=config.mlp_activations,
            intermediate_dropout_rate=config.dropout_rate,
            dtype=config.dtype,
            name='mlp',
            config=config
        )(layer_norm_output, deterministic=deterministic)
        
        mid_output = mlp_output + attention_output
        mid_output_dropped_out = nn.Dropout(
            rate=config.dropout_rate,
            broadcast_dims=(-2,)
        )(mid_output, deterministic=deterministic)
        
        output = mid_output_dropped_out + inputs
        output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'activation_embed'))
        
        if config.scan_layers:
            return output, None
        else:
            return output
        

class Decoder(nn.Module):
    config: Config
    shared_embedding: nn.Module
    
    @nn.compact
    def __call__(self,
                 decoder_input_tokens,
                 decoder_positions=None,
                 decoder_mask=None,
                 deterministic=False,
                 decode=False,
                 max_decode_length=None):
        config = self.config
        
        y = self.shared_embedding(decoder_input_tokens.astype('int32'))
        y = nn.Dropout(
            rate=config.dropout_rate,
            broadcast_dims=(-2,)
        )(y, deterministic=deterministic).astype(config.dtype)
        
        BlockLayer = DecoderLayer
        
        # Gradient checkpointing policy
        if config.remat_policy != 'none':
            if config.remat_policy == 'minimal':
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            elif config.remat_policy == 'proj':
                policy = jax.checkpoint_policies.save_only_these_name('query_proj', 'value_proj', 'key_proj')
            else:
                assert config.remat_policy == 'full', "Remat policy needs to be on list of remat policies"
                policy = None
            
            BlockLayer = nn.remat(
                BlockLayer,
                prevent_cse= not config.scan_layers,
                policy=policy,
                static_argnums=(-1, -2, -3, -4)
            )
        
        # Scan layers
        if config.scan_layers:
            initializing = self.is_mutable_collection('params')
            params_spec = (
                config.param_scan_axis if initializing else nn_partitioning.ScanIn(config.param_scan_axis)
            )
            cache_spec = 0
            y, _ = nn.scan(
                BlockLayer,
                variable_axes={
                    'params': params_spec,
                    'cache': cache_spec,
                    'intermediates': 0
                },
                split_rngs={
                    'params': True,
                    'dropout': config.enable_dropout
                },
                in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
                length=config.num_decoder_layers,
                metadata_params={
                    nn.PARTITION_NAME: 'layers'
                }
            )(
                config=config, name='decoder'
                )(
                    y, decoder_mask,
                    deterministic, decode, max_decode_length
                )
        
        y = LayerNorm(dtype=config.dtype, name='decoder_norm', kernel_axes=('embed',))(y)
        y = nn.Dropout(
            rate=config.dropout_rate,
            broadcast_dims=(-2,)
        )(y, deterministic=deterministic)
        
        # LM Head
        logits = DenseGeneral(
            config.vocab_size,
            dtype=jnp.float32,
            kernel_axes=('embed', 'vocab'),
            name='logits_dense',
            config=config
        )(y)
        logits = nn.with_logical_constraint(logits, ('activation_batch', 'activation_length', 'activation_vocab'))
        
        return logits

class DecoderOnlyTransformer(nn.Module):
    config: Config
    
    def setup(self):
        config = self.config
        self.shared_embedding = Embedding(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            dtype=config.dtype,
            attend_dtype=jnp.float32,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name='token_embedder'
        )
        self.decoder = Decoder(config=config, shared_embedding=self.shared_embedding)
        
    def __call__(
        self,
        decoder_input_tokens,
        decoder_target_tokens,
        decoder_segment_ids=None,
        decoder_positions=None,
        enable_dropout=True,
        decode=False,
        max_decode_length=None
    ):
        config = self.config
        
        if decode:
            decoder_mask = None
        else:
            decoder_mask = make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens,
                dtype=config.dtype,
                decoder_segment_ids=decoder_segment_ids
            )
    
        logits = self.decoder(
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length
        )
        
        return logits