import checkpointing
import functools

import max_logging

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

def log(text):
    print(text, flush = True)
    
def l2norm_pytree(x):
    pass

def activate_profiler(config):
    pass

def deactivate_profiler(config):
    pass

def _prepare_metrics_for_json(metrics, step, run_name):
    pass

def write_metrics_locally(metrics, step, config, file):
    pass

def write_metrics_for_gcs(metrics, step, config, running_metrics):
    pass

def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
    pass

def create_device_mesh(config, logging):
    pass

def unbox_logicaly_partioned_trainstate(boxed_train_state):
    pass

def init_train_state(model, optimizer, config, key):
    pass

def setup_initial_state(model, optimizer, config, rng, mesh, checkpoint_manager):
    pass

def rsqrt_schedule(init_value, shift):
    pass

def create_learning_rate_schedule(learning_rate, warmup_steps):
    pass