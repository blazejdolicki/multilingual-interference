#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=meta_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:00
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
python train_meta.py --inner_lr_decoder 1e-03 --inner_lr_bert 1e-04 --meta_lr_decoder 5e-04 --meta_lr_bert 1e-05 --updates 20 --episodes 500 --support_set_size 20 --model_dir logs/bert_finetune_en/fine_tuned --pairwise 1 --seed 1 --name Korean_Hindi
# python train_meta.py --inner_lr_decoder 1e-03 --inner_lr_bert 1e-04 --meta_lr_decoder 5e-04 --meta_lr_bert 1e-05 --updates 20 --episodes 500 --support_set_size 20 --model_dir logs/bert_finetune_en/fine_tuned --pairwise 2 --name Korean_Arabic
