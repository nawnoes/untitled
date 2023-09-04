from etils import epath
import jax
import portpicker
from jax.experimental import multihost_utils
from orbax import checkpoint
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions, Checkpointer, AsyncCheckpointer
from orbax.checkpoint import type_handlers
import socket

from utils import log

from flax.training import train_state

def _multislice_distribute_initialize():
    """Call jax.distribute.initialize() with appropriate multislice arguments"""
    
    def gen_local_ip():
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    
    def gen_local_ip_nums():
        return [int(num) for num in gen_local_ip().split(':')[-1].split('.')]
    
    def get_coordinator_ip():
        local_ip_nums = jax.numpy.array(gen_local_ip_nums())
        coordinator_ip_nums = multihost_utils.broadcast_one_to_all(local_ip_nums)
        coordinator_ip_strings = [str(num) for num in list(coordinator_ip_nums)]
        
        return '.'.join(coordinator_ip_strings)
    
    port = multihost_utils.broadcast_one_to_all(jax.numpy.array(portpicker.pick_unused_port()))
    coordinator_address = get_coordinator_ip() + ':' + str(port)
    
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=jax.process_count(),
        process_id=jax.process_index()
    )
    
def create_orbax_checkpoint_manager(checkpoint_dir, enable_checkpointing, use_async):
    if not enable_checkpointing:
        log('Checkpointing disabled, not creating checkpoint manager')
        return None
    
    log('Creating checkpoint manager')
    p = epath.Path(checkpoint_dir)
    checkpointer = checkpoint.PyTreeCheckpointHandler()
    
    if use_async:
        _multislice_distribute_initialize()
        checkpointer = AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())
    else:
        checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
        
    manager = CheckpointManager(p, checkpointer, options=CheckpointManagerOptions(create=True))
    log('Checkpoint manager created')
    
    return manager

def load_state_if_possible(checkpoint_manager,
                           first_checkpoint_path,
                           load_from_other_directory,
                           load_from_other_directory_step,
                           abstract_unboxed_pre_state,
                           mesh,
                           state_mesh_annotations):
    if checkpoint_manager is None:
        log('no checkpoint manager, not restoring checkpoint')
        return None, None
    
    def map_to_pspec(data, pspec):
        if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)) and pspec is not None:
            return type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)
        else:
            return type_handlers.RestoreArgs()
        
    restore_args = jax.tree_util.tree_map(map_to_pspec, abstract_unboxed_pre_state, state_mesh_annotations)
    lastest_step = checkpoint_manager.lastest_step()
    
    if lastest_step is not None:
        log(f' Restoring state from this run\'s directory latest step {lastest_step}')
        return checkpoint_manager.restore(lastest_step, abstract_unboxed_pre_state,{"restore_args": restore_args}), None
    
    elif first_checkpoint_path != "":
        log(f'Restoring state from first_checkpoint_path {first_checkpoint_path}')
        p = epath.Path(first_checkpoint_path)
        checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
        return None, checkpointer.restore(p, item=abstract_unboxed_pre_state, restore_args=restore_args).params
    
    elif load_from_other_directory != "":
        p = epath.Path(first_checkpoint_path)
        checkpointer_loader = Checkpointer(checkpoint.PyTreeCheckpointHandler())
        manager_loader = CheckpointManager(p, checkpointer_loader, options=CheckpointManagerOptions(create=True))
        if load_from_other_directory_step == -1:
            step = manager_loader.latest_step()
            log(f'Restoring state from {load_from_other_directory} latest step {step}')
        else:
            step = load_from_other_directory_step
            log(f'Restoring state from {load_from_other_directory} latest step {step}')
        
        return manager_loader.restore(step, abstract_unboxed_pre_state, {'restore_args': restore_args}), None
    else:
        log(f'No existing checkpoints found, not restoring checkpoint')
        return None, None
    