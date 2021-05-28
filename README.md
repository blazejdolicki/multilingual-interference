# Negative Interference in Multilingual Meta-learning
This repository contains code for a project about tackling negative interference in a multilingual meta-learning setup for the task of dependency parsing. 
The codebase we built upon has been generously shared by the authors of the paper [Meta-learning for fast cross-lingual adaptation in dependency parsing](https://arxiv.org/abs/2104.04736) on whose work we build upon. You can find the authors' original readme in `metalearning/original_readme.md`.

Below please find the recipe for reproducing our results:

## Pretrain mBert on English (in Lisa)
After connecting to Lisa, clone this repository into the desired directory.
The `metalearning` folder is our fork from https://github.com/tamuhey/udify/tree/library which is a fork from the original Udify repo. Navigate to that folder (`cd metalearning`). We use that fork because it added support for allennlp>=0.9 which is important because v1 switched from custom allenlp to standard Pytorch data loaders (which should make our further experiments easier).

Load Lisa's conda environment 
```
module load 2019
module load Anaconda3/2018.12
conda activate atcs
```

Install the following dependencies (Python 3.8 seems to be a safe bet):
```
pip install allennlp==1.3.0 tensorflow==2.3.0 pandas jupyter conllu
```
We need tensorflow 2.3 because 2.4 is incompatible with CUDA 10.1 and cuDNN 7.6 which are used by Lisa.

Create the directory for the data in `../multilingual-interference/metalearning`:
```
mkdir -p data/ud-treebanks-v2.3
mkdir -p data/exp-mix
mkdir -p data/concat-exp-mix
```

Navigate back to the `metalearning` directory (`cd ..`) and download the data.
```
bash ./scripts/download_ud_data.sh
```
It seems that `download_ud_data.sh` not only downloads the data but also creates a treebank for all languages.

Run a script that copies treebanks of all languages that Anna used in her paper (based on Table 7). You can run it in the root metalearning directory.
```
python scripts/make_expmix_folder.py
```

Afterward, you can just pass the name of the folder with all these treebanks to concatenate them. `concat_treebanks.py` needs imports Udify's `util.py` which imports stuff like torch, so we need to run `concat_treebanks.py` in a batch script. For that, you can use `concat_treebanks.sh`. Run it from the root directory of metalearning with the command:

```
sbatch concat_treebanks.sh
```

After concatenating treebanks of all relevant languages, create the vocabulary (around 15 minutes):
```
sbatch create_vocabs.sh
```

We copied `config/ud/en/udify_bert_finetune_en_ewt.json` from Anna's repo and changed the vocab directory to the one just created. As described [here](https://github.com/allenai/allennlp/releases/tag/v1.0.0.rc1), we had to change the config file from using an iterator to dataloader.

Run `bert_en_finetune.sh` to finetune mBERT on English. To finetune on Hindi run `bert_en_finetune_hindi.sh`.

## Setup meta-learning and cosine similarity calculation

1. Add pytorch and other libs to env if they weren't added before.
2. Check your unique path to the pre-trained mBERT generated from pretraining. It looks something like `logs/bert_finetune_en/2021.05.12_23.02.00`.
3. Fine-tuning process creates a file `model.tar.gz`. Untar it with `tar -xzf model.tar.gz`
4. Rename the `weights.th` into `best.th` with `mv weights.th best.th` 
5. Navigate to `multilingual-interference/metalearning` and create the directory that will store the gradient conflict calculations: 
``` 
mkdir cos_matrices
``` 
6. Modify `train_meta.sh` to use the correct --model_dir from your pretraining. Change the flags as desired. With default parameters, it takes around 20 hours.
7. Run in Lisa `sbatch train_meta.sh`.
8. The numpy array containing gradient similarities is located in `metalearning/cos_matrices`. It has shape `num_episodes/save_every x num_train_languages x num_train_languages`. The checkpoint gradient similarities are saved  every `save_every` parameter.


### Requirements

After Pretraining mBERT it will generate a path like: `logs/bert_finetune_en/2021.05.12_23.02.00` depending on the date you pretrained it, you will need to pass that path as an argument when running `meta_train.py`.

Make sure you have the necessary requirements `
pip install allennlp==1.3.0 tensorflow==2.3.0 pandas jupyter conllu editdistance learn2learn parsimonious word2number sqlparse`, we use torch  1.7.1 which can be installed using `
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch`


***
## Training
It should be straightforward. Locally in wsl2 or another Linux os run `python train_meta.py --model_dir logs/bert_finetune_en/2021.05.12_23.02.00` indicating the proper part to the pretrained bert. By default `support_set_size` and `episodes` are set to 1 to test if the whole script runs.    

Another example with proper parameters:   
`python train_meta.py --inner_lr_decoder 0.0001 --inner_lr_bert 1e-05 --meta_lr_decoder 0.0007 --meta_lr_bert 1e-05 --updates 20 --episodes 500 --support_set_size 20 --model_dir logs/bert_finetune_en/2021.05.12_23.02.00`

> We still need to double-check which parameters we should use to get the results of the paper, they are explained in the appendix if you wanna use them.

**NOTE:** It is not possible to run the full training with a GPU with less than **24GB** of memory! So when using Lisa we need to use the RTX titan. The job file already uses this (_gpu_titanrtx_shared_course_). Even with GPU equipped with 24GB memory OOM errors might occur! 

### LISA

To run it in lisa use `sbatch train_meta.sh` but first you need to modify the ` --model_dir logs/bert_finetune_en/2021.05.12_23.02.00` to the right path in the `train_meta.sh` file.

### Evaluation and Meta-testing

To do evaluation or Meta-testing we use the script `metatest_all.py`. It will generate a folder like `metavalidation_0.0001_1e-05_20_20_sgd_saved_models-XMAML_0.001_0.001_0.001_0.001_5_9999_1` with the scores in json files.

### Evaluation

Run `python metatest_all.py --validate True  --lr_decoder 1e-03 --lr_bert 1e-04 --updates 20 --support_set_size 20 --optimizer sgd --episode 500 --model_dir saved_models/XMAML_0.001_0.001_0.001_0.001_5_9999`  where the path for `--model_dir` was created after running `train_meta.py` and the filepath corresponds to the params of the run.  _This can be done without the RTX gpu._

### Meta-testing

For this, we will need the tiny-treebanks split for cross-validation. Run `python split_files_tiny_auto.py` and it will take care of making the test files. 
We run the same command as for validation but without the --validate flag. `python metatest_all.py  --lr_decoder 1e-03 --lr_bert 1e-04 --updates 20 --support_set_size 20 --optimizer sgd --episode 500 --model_dir saved_models/XMAML_0.001_0.001_0.001_0.001_5_9999`  
_Need more than 8gb of gpu memory._

We introduced two new flags:

`language_order` - defines the language order in meta-training. Defaults to Anna's (arbitrary?) choice. The two options are 1 and 2, where order 1 is the order in which the most similar languages follow each other. That means that we start with Norwegian because it's the most similar to the pretraining language (English) and then Russian because it is the most similar to Norwegian etc. Order 2 is the opposite, where each language is chosen based on whether they are the least similar to the language before it. The similarities are based on Anna's paper.

`save_every` - how often to save the gradient conflicts. Calculating them is slow because we are operating on seven arrays of length 190M.
