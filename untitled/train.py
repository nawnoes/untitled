import jax
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
print(f"JAX TPU COUNT: {jax.device_count()}")

import datetime
from absl import app
from typing import Sequence
import optax
import numpy as np
from flax.linen import partitioning as nn_partitioning
from tensorboardx import SummaryWriter

from layers import DecoderOnlyTransformer
import hparam
from dataset import get_datasets, preprocess_dataset
import utils
# import temperature_sampler # for predict loop, to be added later
import checkpointing

import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

from utils import log
from train_utils import (
    calculate_training_tflops, 
    get_first_step, 
    load_next_batch, 
    record_scalar_metrics, 
    record_activation_metrics, 
    write_metrics, 
    calculate_num_params_from_pytree
)

cc.initialize_cache(os.path.expanduser("~/.jax_cache"))


def train_step(model, config, state, data, dropout_rng):
    """
    Args:
        model: A nn.Module
        state: A pytree of the current state of the model
        data: Batch of data to apply to the model
        dropout_rng: RNG for dropout layers
    
    Returns:
        new_state: Same format as state
        metrics: Dictionary of model metrics such as loss, training rate, etc
        rng2: A new rng key that can be used in future call
    """
    
    rng1, rng2 = jax.random.split(dropout_rng)

    def loss_fn(params):
        logits, intermediate_outputs = model.apply(
            {'params': params},
            data['inputs'],
            data['targets'],
            data['inputs_segmentation'],
            data['inputs_position'],
            enable_dropout=config.enable_dropout,
            rngs={'dropout':rng1},
            mutable=['itermediates']
            )
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(logits, data['targets'])
        cross_entropy = cross_entropy * (data['input_segmentation'] != 0) # Mask out paddings at the end of each example
        return jnp.sum(cross_entropy) / jnp.size(cross_entropy), intermediate_outputs
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, intermediate_outputs), grads = grad_fn(state.params)
    
    new_state = state.apply_gradients(grads=grads)
    metrics = {'scalar': {'learning/loss': loss, 'learning/grad_nrom': utils.l2norm_pytree(grads),
                          'learning/param_norm': utils.l2norm_pytree(new_state.params)},
               'scalars': {}}
    if config.record_internal_nn_metrics:
        record_activation_metrics(metrics, intermediate_outputs, config)
    
    return new_state, metrics, rng2

def predict_step(inputs, state, rngkey, model, config):
    pass

def train_loop(config, state=None):
    writer = SummaryWriter(config.tensorboard_dir)
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing
    )
    # Init PRNG Keys
    init_rng, next_rng = random.split(random.PRNGKey(0), 2)
    
    # Model & Optimizer
    model = DecoderOnlyTransformer(config)
    learning_rate_scheduler = utils.create_learning_rate_scheduler(
        learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
    )
    
    optimizer = optax.adam(
        utils.create_learning_rate_schedule(
          learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
        ),
        b1 = config.adam_b1,
        b2 = config.adam_b2,
        eps = config.adam_eps,
        eps_root = config.adam_eps_root,  
    )
    
    # Mesh
    devices_array = utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    
    # Datasets
    train_ds, eval_ds = get_datasets(config=config)
    train_iter, _, _, _ = preprocess_dataset(config, mesh, train_ds, eval_ds, vocab_path=os.path.join(config.base_output_directory, config.vocab_relative_path))
    
    # State
    state, state_mesh_annotations = utils.setup_initial_state(model, optimizer, init_rng, mesh, checkpoint_manager)
    data_pspec = P(*config.data_sharding)
    
    # Number of parameters and TFLOPS
    num_model_parameters = calculate_num_params_from_pytree(state.params)
    log(f'Number of model parameters: {num_model_parameters/10**9:.3f} Billion')
    per_device_tflops = calculate_training_tflops(num_model_parameters, config)
    log(f'Per device TFLOPS: {per_device_tflops:.3f}')
    
    # Pjit training step
    p_train_step = pjit(
        train_step,
        in_axis_resources=(state_mesh_annotations, data_pspec, None),
        out_axis_resources=(state_mesh_annotations, None, None),
        static_argnums=(0,1),
        donate_argnums=2
    )
    
    example_batch = None
    for step in np.arage(get_first_step(config), config.steps):
        example_batch = load_next_batch(train_iter, example_batch, config)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            state, metrics, next_rng = p_train_step(
                model, config, state, example_batch, next_rng
            )
        if step > 0 and step % config.save_period == 0 and checkpoint_manager is not None:
            checkpoint_manager.save(step, state)
            log(f'Saved checkpoint | step {step}')
    
    writer.close()
    return state
    
def main(argv: Sequence[str]):
    hparam.initialize(argv)
    os.environ['TFDS_DATA_DIR'] = hparam.config.dataset_path
    train_loop(hparam.config)
    
if __name__ == '__main__':
    app.run(main)



