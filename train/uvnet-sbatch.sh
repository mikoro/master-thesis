#!/bin/bash
#SBATCH --job-name=uvnet-train
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --constraint=k40
#SBATCH -o /wrk/moronkai/slurm-output/%j.out
#SBATCH -e /wrk/moronkai/slurm-output/%j.err

if [ -z "$1" ]; then
    echo "Please give the target git tag name as the first argument"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please give the target CUDA GPU device id as the second argument"
    exit 1
fi

module load cuda/8.0

TAG_NAME=$1
CUDA_DEVICE_ID=$2
PARAMS_FILE=$3
CLONE_DIR=master-thesis-clones/master-thesis-${SLURM_JOB_ID}-${TAG_NAME}
CODE_DIR=${WRKDIR}/${CLONE_DIR}/code

export PATH=${WRKDIR}/anaconda3/bin:${WRKDIR}/openmpi/bin:${PATH}
export LD_LIBRARY_PATH=${WRKDIR}/glibc/lib:${WRKDIR}/openmpi/lib:${LD_LIBRARY_PATH}
export TMPDIR=/tmp/moronkai/${SLURM_JOB_ID}

mkdir -p ${TMPDIR}

echo "Extracting train and test images"
time tar -xf ${WRKDIR}/renders/head_final7.tar -C ${TMPDIR}

echo "Cloning the git repository"
cd ${WRKDIR}
git clone master-thesis ${CLONE_DIR}
cd ${CLONE_DIR}
git config remote.origin.url https://mikoro@bitbucket.org/aaltogfx/master-thesis-mikko.git
git fetch -v
git fetch --tags -v
git checkout tags/${TAG_NAME}

echo "Running the training script"

umask u=rwx,go=rx

if [ -z "${PARAMS_FILE}" ]; then
    srun --gres=gpu:1 python ${CODE_DIR}/uvnet.py --gpu=${CUDA_DEVICE_ID}
else
    srun --gres=gpu:1 python ${CODE_DIR}/uvnet.py --gpu=${CUDA_DEVICE_ID} --params=${PARAMS_FILE}
fi

used_slurm_resources.bash
