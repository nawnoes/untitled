"""Script to run a command in a multislice/multihost environment

The "runner" host (the one which runs this script) and the "worker" hosts (the TPUS found by --TPU_PREFIX and where the
--COMMAND is run) should be different. You can use either a TPUVM or a non-TPUVM runner host,
but in the former case this host needs to be in the same project as the worker hosts,
but not be one of the workers.

Example usages:
    Assuming runner.sh lives in directory path/to/dir
    python3 multihost_run.py --TPU_PREFIX=mytpu-name --COMMAND="bash runner.sh" --SCRIPT_DIR=path/to/dir
        this will recursively copy all of the files in path/to/dir to each tpu and run runner.sh

Common issues:
    Make sure your gcloud zone in set with e.g.
        gcloud config set compute/zone us-central2-b
        gcloud config set project <project_name>
        before running this script.
        
    You may have to create/authorize ssh-keys when first sshing into TPUs.
    For this purpose you may need to first run:
        ssh-keygen -f ~/.ssh/google_compute_engine
"""

import argparse
import sys
from collections import namedtuple
import subprocess
import time
from datetime import datetime
import os
import re

def get_project():
    completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
    project_output = completed_command.stdout.decode().strip().split('\n')
    
    if len(project_output) < 1 or project_output[-1] == '':
        sys.exit('You must specify the project in the PROJECT flag or set it with "gcloud config set project <project>"')
    return project_output[-1] # the project name lives on the last line of the output

def get_zone():
    completed_command = subprocess.run(['glcoud', 'config', 'get', 'compute/zone'], check=True, capture_output=True)
    zone_outputs = completed_command.stdout.decode().strip().split('\n')
    if len(zone_outputs) < 1 or zone_outputs[-1] == '':
        sys.exit('You must specify the zone in the ZONE flag or set it with "gcloud config set compute/zone <zone>"')
    return zone_outputs[-1]

def default_run_name():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser(description='TPU configuration options')
parser.add_argument('--TPU_PREFIX', type=str, default=None, required=True, 
                    help='prefix of worker tpus. if tpus are name user-0 and user-1, TPU_PREFIX should be set as user')
parser.add_argument('--PROJECT', type=str, default=None,
                    help='GCE project name, default to gcloud config project')
parser.add_argument('--ZONE', type=str, default=None, 
                    help='GCE ZOne, e.g. us-central2-b, defaults to gcloud config compute/zone')
parser.add_argument('--SCRIPT_DIR', type=str, default=os.getcwd(), 
                    help='The local location of the directory to copy to the TPUs and run the main command from defaults to current working directory')
parser.add_argument('--COMMAND', type=str, default=None, required=True, 
                    help='Main command to run on each TPU. this command is run from a copied version of SCRIPT_DIR on each TPU worker.')
parser.add_argument('--RUN_NAME', type=str, default=default_run_name(),  
                    help='NAme for the code directory on the TPU')
parser.add_argument('--USE_EXISTING_FOLDER', type=str, default='False', 
                    help='If true, use the exising code directory on the TPU')
parser.add_argument('--INTERNAL_IP', type=str, default='False', 
                    help='Set true if running script locally from a TPU or GCE instance, false otherwise.')

args = parser.parse_args()
args.USE_EXISTING_FOLDER = args.USE_EXISTING_FOLDER.lower() == "true"
args.INTERNAL_IP = args.INTERNAL_IP.lower() == "true"

Slice = namedtuple('Slice', ['name', 'slice_num', 'num_workers', 'version'])

def get_slices():
    """Retruns a list of slices matching TPU_PREFIX"""
    command = [
        'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'list',
        f'--filter=name~{args.TPU_PREFIX}', '--format=csv(name,accelerator_type)',
        f'--project={args.PROJECT}', f'--zone={args.ZONE}'
    ]
    try:
        completed_command = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred trying to find TPU slices named {args.TPU_PREFIX} or matching regex {args.TPU_PREFIX}-[0-9]+ in project {args.PROJECT} zone {args.ZONE}')
        print(f'Error is:\n {e.stderr}')
        return []
    
    instances = completed_command.stdout.decode()
    instance_list = instances.strip().split('\n')
    instance_list = filter_instances(instance_list[1:], args.TPU_PREFIX) # First row is headers
    
    num_slices = len(instance_list)
    slices = [None for _ in range(num_slices)]
    
    if num_slices > 0:
        print(f'{num_slices} slices found.', flush=True)
    else:
        print(f'No TPUs found with name {args.TPU_PREFIX} or matching regex {args.TPU_PREFIX}-[0-9]+ in project {args.PROJECT} and zone {args.ZONE}.')
        return []
    
    slice_names = [instance.split(',')[0] for instance in instance_list]
    slice_versions = [instance.split(',')[1] for instance in instance_list]
    
    # Get number of workers in any slice (assume same worker count for all slice)
    command = [
        'gcloud', 'compute', 'tpus', 'describe', slice_name[0],
        '--flatten=networkEndpoints[]', '--format=csv[no-heading](networkEndpoint.ipAddress)',
        f'--project={args.PROJECT}', f'--zone={args.ZONE}'
    ]
    completed_command = subprocess.run(command, capture_output=True, check=True)
    num_workers = len(completed_command.stdout.decode().strip().split('\n'))
    
    for slice_name, version in zip(slice_names, slice_versions):
        if num_slices > 1:
            slice_num = int(slice_name.split('-')[-1])
        else:
            slice_num = 0
        
        slices[slice_num] = Slice(slice_name, slice_num, num_workers, version)
    
    return slices

def filter_instances(instance_list, tpu_prefix):
    # First look for exact match with tpu_prefix
    for instance in instance_list:
        if instance.split(',')[0] == tpu_prefix:
            return [instance]
    
    # If no exact match, reg-exp full match "<tpu_prefix>-[0-9]+"
    re_pattern = tpu_prefix + '-[0-9]+'
    return [instance for instance in instance_list if re.fullmatch(re_pattern, instance.split(',')[0])]
    
def write_kill_script(kill_processes_script_name):
    kill_processes_script = os.path.join(args.SCRIPT_DIR, kill_processes_script_name)
    
    with open(kill_processes_script, 'w', encoding='utf-8') as f:
        f.write(kill_existing_processes_str())

def kill_existing_processes_str():
    commond = """
#!/bin/bash
_TPU_VERSION_NAME="${1}"
device_name="accel"
if [[ "${_TPU_VERSION_NAME}" =~ ^v5.* ]]; then
    device_name="vfio/"
fi

echo "Searching for existing processes on device ${device_name}..."
pids=$(sudo lsof -t /dev/${device_name}* | sort -u)
if [[ -n "${pids}" ]]; then
    echo "Existing processed found with pid ${pids}"
    for pid in ${pids}; do
        echo "Killing process ${pid}..."
        kill -9 "${pid}"
        tail --pid="${pid}" -f /dev/null
        echo "Existing process %{pid} on your TPU was killed successfully!"
    done
    echo "All existing processes killed, so your TPU is ready to use!"
else
    echo "No existing process found, so your TPU is ready to use!"
fi
sudo rm -f /tmp/libtpu_lockfile"""
    return commond

def scps(slices, run_name_dir, zip_name):
    """Zip the script directory, scp it to the TPUs, and unzip it there"""
    original_working_directory = os.getcwd()
    os.chdir(args.SCRIPT_DIR) # To tar script_dir, it is most convenient to cd there.
    
    # Zip script directory
    # Save the zip both to the logging directory, and the script directory.
    # It will be removed from the script directory after the transfer to the TPUs
    os.makedirs(run_name_dir, exist_ok=True)
    zip_path = os.path.join(run_name_dir, zip_name)
    command = ['tar', '--exclude=tmp', '-czf', zip_path, './']
    subprocess.run(command, check=True)
    
    # Move zip file to each tpuvm worker
    commands = []
    worker_list = []
    
    for cur_slice in slices:
        for worker_num in range(cur_slice.num_workers):
            command = [
                'gcloud', 'compute', 'tpus', 'tpu-vm', 'scp', f'--worker={worker_num}', zip_path,
                f'{cur_slice.name}:~/', '--strict-host-key-checking=no', f'--project={args.PROJECT}', f'--zone={args.ZONE}'
            ]
            if args.INTERNAL_IP:
                command.append('--internal-ip')
            commands.append(command)
            worker_list.append([cur_slice.slice_num, worker_num])
        
    return_code, _ = run_commands(commands, 0, "SCP", worker_list)
    if return_code != 0:
        print('Failed to scp zipped code directory with error code.', return_code)
        return return_code
    
    # Cleanup
    os.chdir(original_working_directory)
    
    return return_code

def execute_main_command(main_command, slices, local_log_dir, zip_name):
    """Run the main command on each worker, logging each seperately."""
    kill_script_name = "kill_existring_processes.sh" # File written on worker machines
    commands = []
    output_logs = []
    worker_list = []
    os.makedirs(local_log_dir, exist_ok=True)
    
    for slice_num, cur_slice in enumerate(slices):
        for worker_num in range(cur_slice.num_workers):
            output_filename = os.path.join(local_log_dir, f'output_slice_{cur_slice.slice_num:04d}_worker_{worker_num:04d}.txt')
            output_logs.append(output_filename)
            
            mkdir_command = f'mkdir -p {args.RUN_NAME}'
            mv_zip_command = f'mv {zip_name} {args.RUN_NAME}'
            cd_command = f'cd {args.RUN_NAME}'
            unzip_command = f'tar xzf {zip_name}'
            write_kill_script_command = f'echo \'{kill_existing_processes_str()}\' > {kill_script_name}'
            kill_existing_command = f'bash {kill_script_name} {cur_slice.version}'
            
            if args.USE_EXISTING_FOLDER is False:
                remote_command_list = [
                    mkdir_command, mv_zip_command, cd_command, unzip_command,
                    write_kill_script_command, kill_existing_command, main_command
                ]
            else:
                remote_command_list = [cd_command, write_kill_script_command, kill_existing_command, main_command]
            
            remote_command_list_str = " && ".join(remote_command_list)
            
            gcloud_command = [
                'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh', cur_slice.name,
                f'--worker={worker_num}', '--command', remote_command_list_str, '--strict-host-key-checking=no',
                f'--project={args.PROJECT}', f'--zone={args.ZONE}'
            ]
            if args.INTERNAL_IP:
                gcloud_command.append('--internal-ip')
                
            commands.append(gcloud_command)
            worker_list.append([slice_num, worker_num])
    
    return_code, return_codes = run_commands(commands, 0, 'MAIN COMMAND', worker_list, output_logs=output_logs)
    if return_code > 0:
        failure_index = next((i for i, x in enumerate(return_codes) if x), None)
        print(f'Main command failed on slice {worker_list[failure_index[0]]} worker'\
            f' {worker_list[failure_index][1]} with error code {return_codes[failure_index]}, see logs for details', flush=True)
    
    return return_code

def run_commands(commands, id_to_print, jobname, worker_list, is_shell=False, output_logs=None, fail_fast=True):
    pass

def assert_script_dir_exists(script_dir):
    pass

class Tee:
    pass

def main():
    pass

if __name__ == '__main__':
    main()