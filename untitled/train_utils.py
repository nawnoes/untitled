import jax
from utils import log


def calculate_training_tflops(num_model_parameters, config):
    learnable_weight_tflops = 6 * num_model_parameters * config.max_target_length\
        * config.per_devie_batch_size / 1e12
    
    attention_tflops = 12 * config.num_heads * config.num_decoder_layers * config.head_dim\
        * config.max_target_length ** 2 * config.per_devie_batch_size / 1e12

    total_tflops = learnable_weight_tflops + attention_tflops
    
    log(f"Learnable weight TFLOPS: {learnable_weight_tflops:.2f}")
    log(f"Attention TFLOPS       : {attention_tflops:.2f}")
    log(f"Total TFLOPS           : {total_tflops:.2f}")

def get_first_step(state):
    with jax.spmd_mode('allow_all'):
        return int(state.step)
    
def load_next_batch(train_iter, example_batch, config):
    if config.reuse_example_batch and example_batch is not None:
        return example_batch
    else:
        train_iter()
        
def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr):
    metrics['scalar'].update({
        'perf/step_time_seconds': step_time_delta.total_seconds(),
    })
    metrics['scalar'].update({
        'perf/per_device_tflops': per_device_tflops,
    })
    metrics['scalar'].update({
        'perf/per_device_tflops_per_second': per_device_tflops / step_time_delta.total_seconds()
    })
    metrics['scalar'].update({'learning/current_learning_rate': lr })
    
def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """ Adds the activation metrics to the metrics dict"""
  if config.scan_layers:
    metrics_dict = intermediate_outputs['intermediates']['decoder']['decoder']

    for layer_num in range(config.num_decoder_layers):
      output_metrics['scalar'][f'activ_fraction_zero/layer_{layer_num:03d}'] = metrics_dict["activation_fraction_zero"][0][layer_num]
      output_metrics['scalar'][f'activ_mean/layer_{layer_num:03d}'] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics['scalar'][f'activ_stdev/layer_{layer_num:03d}'] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs['intermediates']['decoder'][f'layers_{layer_num}']
      output_metrics['scalar'][f'activ_fraction_zero/layer_{layer_num:03d}'] = layer["activation_fraction_zero"][0]
      output_metrics['scalar'][f'activ_mean/layer_{layer_num:03d}'] = layer["activation_mean"][0]
      output_metrics['scalar'][f'activ_stdev/layer_{layer_num:03d}'] = layer["activation_stdev"][0]

def write_metrics(writer, metrics, step, config):
    with jax.spmd_mode('all_all'):
        if jax.process_index() == 0:
            for metric_name in metrics.get("scalar", []):
                writer.add_scalar(metric_name, metrics["scalar"][metric_name], step)
    
        full_log = step % config.log_period == 0
        
        log(f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
            f"TFLOP/s       : {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
            f"loss          : {metrics['scalar']['learning/loss']:.3f}")
        
        if full_log:
            log(f'To see full metrics \'tensorboard --logdir={config.tensorboard_dir}\'')
            writer.flush()
        
def calculate_num_params_from_pytree(params):
    params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
    total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
    return total_parameters