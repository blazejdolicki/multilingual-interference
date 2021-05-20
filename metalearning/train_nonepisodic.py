"""This code performs multi-task learning in a non-episodic manner"""
import json
import subprocess
import argparse
import os
import torch
from torch import autograd
import numpy as np
from torch.optim import Adam

from allennlp.models.model import Model
from allennlp.models.archival import archive_model
from allennlp.common.util import prepare_environment
from udify import util
import naming_conventions

from get_language_dataset import get_language_dataset
from get_default_params import get_params
from naming_conventions import train_languages, train_languages_lowercase
from schedulers import get_cosine_schedule_with_warmup
from allennlp.nn.util import move_to_device
from sklearn.metrics.pairwise import cosine_similarity

import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument("--lr_decoder", default=5e-4, type=float, help="Adaptation LR")
    parser.add_argument("--lr_bert", default=1e-5 , type=float, help="Adaptation LR for BERT layers" )
    parser.add_argument("--episodes", default=500, type=int, help="Episode amount")
    parser.add_argument("--model_dir", default=None, type=str, help="Directory from which to start training", )
    parser.add_argument("--support_set_size", default=None, type=int, help="Batch size")
    parser.add_argument("--name", default="non_episodic", type=str, help="Extra name")

    parser.add_argument("--addenglish", default=False, type=bool, help="Add English as a task")
    parser.add_argument("--notaddhindi", default=False, type=bool, help="Add English as a task" )
    parser.add_argument("--save_every", default=4, type=int, help="Save the gradient conflicts every save_every episodes ")
    parser.add_argument("--language_order", default=0, type=int, help="The order of languages in the inner loop")
    parser.add_argument("--accumulation_mode", default="sum", type=str, help="What gradient accumulation strategy to use", choices=["mean", "sum"])

    args = parser.parse_args()

    training_tasks = []
    device = torch.device('cuda')
    # 7 languages by default -R
    if args.language_order == 1:
        lan_ = naming_conventions.train_languages_order_1
        lan_lowercase_ = naming_conventions.train_languages_order_1_lowercase

    elif args.language_order == 2:
        lan_ = naming_conventions.train_languages_order_2
        lan_lowercase_ = naming_conventions.train_languages_order_2_lowercase

    else:
        lan_ = naming_conventions.train_languages
        lan_lowercase_ = naming_conventions.train_languages_lowercase

    for lan, lan_l in zip(lan_, lan_lowercase_):
        if not ("indi" in lan and args.notaddhindi):
            training_tasks.append(get_language_dataset(lan, lan_l, seed=args.seed, support_set_size=args.support_set_size))

   
    if args.addenglish:
        training_tasks.append(get_language_dataset("UD_English-EWT", "en_ewt-ud", seed=args.seed, support_set_size=args.support_set_size))
    
    train_params = get_params("finetuning", args.seed)
    prepare_environment(train_params)

    EPISODES = args.episodes
    LR_DECODER = args.lr_decoder
    LR_BERT = args.lr_bert
    PRETRAIN_LAN = args.model_dir.split("/")[-2] # says what language we are using

    MODEL_SAVE_NAME = (
        "saved_models/finetune_"
        + str(LR_DECODER)
        + "_"
        + str(LR_BERT)
        + "_"
        + str(args.seed)
    )
    MODEL_VAL_DIR = MODEL_SAVE_NAME + args.name
    MODEL_FILE = (args.model_dir if args.model_dir is not None else "logs/bert_finetune_en/2021.05.13_01.56.30")
    
    if not os.path.exists(MODEL_VAL_DIR):
        subprocess.run(["mkdir", MODEL_VAL_DIR])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
        subprocess.run(["cp", "-r", MODEL_FILE + "/vocabulary", MODEL_VAL_DIR])
        subprocess.run(["cp", MODEL_FILE + "/config.json", MODEL_VAL_DIR])

    model = Model.load(train_params, MODEL_FILE).cuda()
    model.train()

    optimizer = Adam(
        [
            {"params": model.text_field_embedder.parameters(), "lr": LR_BERT},
            {"params": model.decoders.parameters(), "lr": LR_DECODER},
            {"params": model.scalar_mix.parameters(), "lr": LR_DECODER},
        ],
        LR_DECODER,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, 1000)
    def restart_iter(task_generator_, args_):
        """ Restart the iter(Dataloader) by creating again the dataset.
        This method is called when we looped through the whole data and want to start from the beginning.
        """       
        # Get the path to datat like: data/ud-treebanks-v2.3/UD_Arabic-PADT/ar_padt-ud-train.conllu
        task_split = task_generator_._dataset._file_path.split('/')
        language = task_split[-2]  # Language in capitals
        # Language in lower case, remove -train.collu, -dev.conllu, -test.conllu
        language_lowercase_ = task_split[-1].split('-')[0]+'-ud'
        return get_language_dataset(language, language_lowercase_, seed=args_.seed, support_set_size=args_.support_set_size)

    with open(MODEL_VAL_DIR + "/losses.txt", "w") as f:
        f.write("model ready\n")
    losses = []
    episode_grads = []  # NI store the gradients of an episode for all languages
    cos_matrices = []
    for episode in range(EPISODES):
        for j, task in enumerate(training_tasks):
            language_grads = torch.Tensor()            
            try:
                    input_set = next(task)

            except StopIteration:
                training_tasks[j] = restart_iter(task, args)
                task = training_tasks[j]
                input_set = next(task)  # Sample from new iter
            input_set = move_to_device(input_set, device)
            loss = model(**input_set)["loss"]
            # task_num_tokens_seen[j] += len(input_set['tokens']['tokens'][0])
            grads = autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=True, allow_unused=True)
            if (episode+1) % args.save_every == 0:  # NI
                        new_grads = [g.detach().cpu().reshape(-1) for g in grads if type(g) == torch.Tensor]  # filters out None grads
                        grads_to_save = torch.hstack(new_grads).detach().cpu()  # getting all the parameters
                        language_grads = torch.cat([language_grads.cpu(), grads_to_save], dim=-1)  # Updates * grad_len in the last update

                        del grads_to_save
                        del new_grads
                        torch.cuda.empty_cache()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (episode+1) % args.save_every == 0:  # NI
                language_grads = language_grads.reshape(-1, 1)  # setup for taking the average

                if args.accumulation_mode == "mean":
                    language_grads = torch.mean(language_grads, dim=1)  # number of gradients x 1
                else:
                    language_grads = torch.sum(language_grads, dim=1)  # number of gradients x 1
                
                episode_grads.append(language_grads.detach().cpu().numpy())
           
        if (episode + 1) % args.save_every == 0:
            epi_grads = np.array(episode_grads)
            print("[INFO]: Calculating cosine similarity matrix ...")
            cos_matrix = cosine_similarity(epi_grads)
            cos_matrices.append(np.array(cos_matrix))
            print("Cos matrices shape", np.array(cos_matrices).shape)

        if (episode + 1) % args.save_every == 0:

            for filename in glob.glob(os.path.join(MODEL_VAL_DIR, "model*")):  # remove the previous temp grads
                os.remove(filename)

            backup_path = os.path.join(MODEL_VAL_DIR, "model" + str(episode + 1) + ".th")
            torch.save(model.state_dict(), backup_path)
            last_iter = episode + 1
        # NI - Save the gradients in case OOM occurs
        if (episode+1) % args.save_every == 0:  # not to slow down a lot

            # Delete the last temp file
            for filename in glob.glob(f"./cos_matrices/temp_allGrads_episode_nonepisodic_pretrain{PRETRAIN_LAN}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}*"): # remove the previoustemp grads
                os.remove(filename) 

            np.save(f"cos_matrices/temp_allGrads_episode_nonepisodic_pretrain{PRETRAIN_LAN}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}_cos_mat{episode}", np.array(cos_matrices))
            torch.cuda.empty_cache()
     # Delete the last temp file
    for filename in glob.glob(f"./cos_matrices/temp_allGrads_episode_nonepisodic_pretrain{PRETRAIN_LAN}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}*"): # remove the previoustemp grads
        os.remove(filename) 

    cos_matrices = np.array(cos_matrices)
    print(f"[INFO]: Saving the similarity matrix with shape {cos_matrices.shape}")
    np.save(f"cos_matrices/allGrads_episode_nonepisodic_pretrain{PRETRAIN_LAN}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}_cos_mat{EPISODES}", cos_matrices)

    for x in losses:
        with open(MODEL_VAL_DIR + "/losses.txt", "a") as f:
            f.write(str(x))
            f.write("\n")
    for i in [EPISODES]:

        filename = os.path.join(MODEL_VAL_DIR, "model" + str(i) + ".th")
        if os.path.exists(filename):
            save_place = MODEL_VAL_DIR + "/" + str(i)
            subprocess.run(["mv", filename, MODEL_VAL_DIR + "/best.th"])
            subprocess.run(["mkdir", save_place])
            archive_model(MODEL_VAL_DIR, archive_path=save_place)
   
    # subprocess.run(["rm", MODEL_VAL_DIR + "/best.th"])


if __name__ == "__main__":
    main()
