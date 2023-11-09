import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
import sys
sys.path.append('/home/dev/untitled/')
sys.path.append('/home/dev/untitled/untitled')

import jax
from untitled.dataset import get_datasets
import hparam

import checkpointing as checkpointing
from absl import app
from typing import Sequence
import optax
import numpy as np
from flax.linen import partitioning as nn_partitioning
from tensorboardX import SummaryWriter

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

from log import log

print(f'devices:{jax.devices()}')
print(f'local_devices:{jax.local_devices()}')

init_rng, next_rng = random.split(random.PRNGKey(0), 2)

hparam.initialize(['','/home/dev/untitled/untitled/config/mini.yml', 'run_name=test_config'])

# Tensorboard summary writer
writer = SummaryWriter(hparam.config.tensorboard_dir)

# orbax에서 checkpoint manager를 초기화
checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        hparam.config.checkpoint_dir,
        hparam.config.enable_checkpointing,
        hparam.config.async_checkpointing
    )

devices_array = utils.create_device_mesh(hparam.config, None)
mesh = Mesh(devices_array, hparam.config.mesh_axes)
data_pspec = P(*hparam.config.data_sharding)

optimizer = optax.adam(
        utils.create_learning_rate_schedule(
          learning_rate=hparam.config.learning_rate, warmup_steps=hparam.config.warmup_steps
        ),
        b1 = hparam.config.adam_b1,
        b2 = hparam.config.adam_b2,
        eps = hparam.config.adam_eps,
        eps_root = hparam.config.adam_eps_root,  
    )

model = DecoderOnlyTransformer(hparam.config)

# State
state, state_mesh_annotations = utils.setup_initial_state(model, optimizer, hparam.config, init_rng, mesh, checkpoint_manager)
print(state)
print(state_mesh_annotations)

print(f'State')