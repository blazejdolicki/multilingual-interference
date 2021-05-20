#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=meta_nonepisodic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=22:59:00
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
#python train_meta.py --model_dir logs/bert_finetune_en/2021.05.15_11.24.02
python train_nonepisodic.py --lr_decoder 1e-04 --lr_bert 7e-06 --episodes 500 --support_set_size 20 --addenglish True --notaddhindi True --model_dir logs/bert_finetune_hindi/2021.05.18_14.46.31