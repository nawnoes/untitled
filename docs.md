### Running on NVIDIA GPUs
1. Clone MaxText.
2. Pull a base container from NVIDIA's [JAX-Toolbox](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-toolbox)
3. Within the root directory of that `git` repo, install dependencies by running:
```
bash setup.sh DEVICE=gpu
```
4. After installation completes, run training with the command:
```
python3 MaxText/train.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```
5. If you want to decode, you can decode as follows.
```
python3 MaxText/decode.py MaxText/configs/base.yml run_name=$YOUR_JOB_NAME
```