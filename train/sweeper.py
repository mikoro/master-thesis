# coding=utf-8

"""Script for generating hyperparameter sweeps. Sends jobs to slurm."""

# This script needs to be run on an interactive GPU node on taito-gpu:
# srun -pgpu --gres=gpu:1 --pty $SHELL
# LD_LIBRARY_PATH needs to be also set:
# export LD_LIBRARY_PATH=${WRKDIR}/glibc/lib:${WRKDIR}/openmpi/lib:${LD_LIBRARY_PATH}

from uvnet_utils import *

wrk_dir = os.environ["WRKDIR"]
assert len(wrk_dir) > 0, "WRKDIR environment variable should be set"
params_dir = os.path.join(wrk_dir, "params")

for i in range(10):
    params = Parameters()
    params.train_minibatch_size += i
    params_file_path = os.path.join(params_dir, "uvnet_params_{0}.dat".format(get_random_string(16)))
    params.save_to_file(params_file_path)
    git_tag_name = "net-c1"
    gpu_device_id = 0
    output = subprocess.check_output(["sbatch", "uvnet-sbatch.sh", git_tag_name, str(gpu_device_id), params_file_path]).decode(sys.stdout.encoding).strip()
    print(output)
