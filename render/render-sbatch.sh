#!/bin/bash
#SBATCH --job-name=head-render
#SBATCH --partition=serial
#SBATCH --time=14:05:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=3000
#SBATCH --array=1-20
#SBATCH -o /wrk/moronkai/slurm-output/%j.out
#SBATCH -e /wrk/moronkai/slurm-output/%j.err

RENDER_NAME=head_final11

export TMPDIR=/tmp/moronkai/${SLURM_JOB_ID}

mkdir -p ${TMPDIR}/output
mkdir -p ${WRKDIR}/renders/${RENDER_NAME}

echo "Extracting textures"
time tar -xf ${WRKDIR}/misc/textures.tar -C ${TMPDIR}

echo "Running the rendering script"
umask u=rwx,go=rx
srun sh ${WRKDIR}/master-thesis/blender/head/render-launch.sh

echo "Tarring the results"
cd ${TMPDIR}/output && tar -cf ../${RENDER_NAME}_${SLURM_JOB_ID}.tar .
mv ../${RENDER_NAME}_${SLURM_JOB_ID}.tar ${WRKDIR}/renders/${RENDER_NAME}

used_slurm_resources.bash
