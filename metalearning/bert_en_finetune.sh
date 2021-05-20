#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=ConcatTreebanks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out


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

# finetune mBERT model with English data using the vocabulary specified in config (that was created from all exp-mix languages)
python train.py --config config/ud/en/udify_bert_finetune_en_ewt.json --name bert_finetune_en
