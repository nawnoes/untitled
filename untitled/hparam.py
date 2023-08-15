from collections import OrderedDict

import math
import os
import sys 
import yaml
import jax

def string_to_bool(s:str):
    """Converts a string to a boolean."""
    if s.lower() in ['true', '1']:
        return True
    elif s.lower() in ['false', '0']:
        return False

_config = None
conifg = None

def _lists_to_tuples(l):
    """Converts lists to tuples recursively.""" 
    return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l

class _HyperParameters():
    def __init__(self, argv, **kwargs):
        with open(argv[1], "r", encoding="utf-8") as yaml_file:
            raw_data_from_yaml = yaml.safe_load(yaml_file)
            
        raw_data_from_cmd_line = self._load_kwargs(argv, **kwargs)
        
        for k in raw_data_from_cmd_line:
            if k not in raw_data_from_yaml:
                raise ValueError(f"Unknown parameter {k}")
        
        raw_keys = OrderedDict()
        for k in raw_data_from_yaml:
            if k not in raw_data_from_cmd_line:
                raw_keys[k] = raw_data_from_cmd_line[k]
            else:
                raw_keys[k] = raw_data_from_yaml[k]
        
        _HyperParameters.user_init(raw_keys)
        self.keys = raw_keys
    
    def _load_kwargs(self, argv, **kwargs):
        args_dict = dict(a.split("=") for a in argv[2:])
        args_dict.update(kwargs)
        return args_dict
    
    @staticmethod
    def user_init(raw_keys):
        raw_keys["dtype"] = jax.numpy.dtype(raw_keys["dtype"])
        run_name = raw_keys["run_name"]
        
        assert run_name, "Erroring out, need a real run_name"
        base_output_directory = raw_keys["base_output_directory"]
        assert base_output_directory, "Erroring out, please set a real base_output_directory"
        
        assert raw_keys['dataset_path'], "Erroring out, please set a real dataset_path in configs/base.yml\n\
        See instructions for downloading the c4 dataset here:\n\
        https://github.com/google/maxtext/blob/main/README.md#getting-started-download-dataset-and-configure.\n"
        
        raw_keys["tensorboard_dir"] = os.path.join(base_output_directory, run_name, "tensorboard", "")
        raw_keys["checkpoint_dir"] = os.path.join(base_output_directory, run_name, "checkpoints", "")
        raw_keys["metrics_dir"] = os.path.join(base_output_directory, run_name, "metrics", "")
        
        raw_keys["logical_axis_rules"] = _lists_to_tuples(raw_keys["logical_axis_rules"])
        raw_keys["data_sharding"] = _lists_to_tuples(raw_keys["data_sharding"])

        emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(raw_keys['global_parameter_scale'])
        raw_keys['emb_dim'] = emb_scale * raw_keys['base_emb_dim']
        raw_keys['num_heads'] = num_head_scale * raw_keys['base_num_heads']
        raw_keys['mlp_dim'] = mlp_dim_scale * raw_keys['base_mlp_dim']
        raw_keys['num_decoder_layers'] = layer_scale * raw_keys['base_num_decoder_layers']


def get_individual_scales(scale):
    log_2_scale = math.floor((math.log2(scale)))
    if 2**log_2_scale != scale:
        raise ValueError("Global parameter scale should be a power of 2. If you want finer grained control of the model sizes "
        "then you can explicitly set base_embed_dim, base_num_heads, base_mlp_dim, base_num_decoder_layers and/or head_dim.")
    base_scale, rem = divmod(log_2_scale, 3)
    base_scale += 1
    emb_scale = base_scale + int(rem > 0)
    num_head_scale = base_scale + int(rem > 1)
    mlp_dim_scale = num_head_scale
    layer_scale = base_scale
    return emb_scale, num_head_scale, mlp_dim_scale, layer_scale


class HyperParameters():
    def __init__(self):
        pass
    
    def __getattr__(self, attr):
        if attr not in _config.keys:
            raise AttributeError(f"Unknown attribute {attr}")
        return _config.keys[attr]
    
    def __setattr__(self, attr, value):
        raise ValueError
    
    
def initialize(argv, **kwargs):
    global _config, config
    _config = _HyperParameters(argv, **kwargs) 
    config = HyperParameters()


if __name__ == "__main__":
    initialize(sys.argv)
    print(config.steps)
    r = range(config.steps)