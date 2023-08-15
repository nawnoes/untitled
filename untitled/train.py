import jax
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
print(f"JAX TPU COUNT: {jax.device_count()}")

import datetime
from absl import app
from typing import Sequesce
import optax
import numpy as np
from flax.linen import partitioning as nn_partitioning
from tensorboardx import SummaryWriter

from utils import log
from layers import DecoderOnlyTransformer
import configs


