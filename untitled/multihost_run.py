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