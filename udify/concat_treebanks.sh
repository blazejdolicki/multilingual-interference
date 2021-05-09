#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=ConcatTreebanks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out

# for some reason we need CUDA here

module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

# Your job starts in the directory where you call sbatch

# Activate your environment
source activate atcs-project

OUTPUT_DIR="data/concat-exp-mix"
mkdir -p $OUTPUT_DIR

# Run your code
srun python concat_treebanks.py $OUTPUT_DIR --dataset_dir data/exp-mix
