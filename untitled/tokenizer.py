import os
import tempfile
import time
from typing import Any, Dict, Iterable, Tuple

import dataclasses
from absl import logging
import tensorflow as tf
import tensorflow_text as tftxt
from sentencepiece import SentencePieceTrainer
import jax

def load_sentencepiece_tokenizer(model_path):
    with tf.io.gfile.GFile(model_path, 'rb') as model_fp:
        sp_model = model_fp.read()
    
    sp_tokenizer = tftxt.SentencepieceTokenizer(model=sp_model)
    
    return sp_tokenizer

@dataclasses.dataclass
class TokenizerOp:
    sp_tokenizer: Any
    data_keys: Iterable[str] = ('inputs', 'targets')
    
    def __call__(self, featrues):
        for k in self.data_keys:
            featrues[k] = self.sp_tokenizer.tokenize(featrues[k])
        return featrues