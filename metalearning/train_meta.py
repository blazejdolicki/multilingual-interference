# -*- coding: utf-8 -*-
"""
This file Meta-Trains on 7 languages
And validates on Bulgarian
"""
#from _typeshed import NoneType
from naming_conventions import train_languages, train_languages_lowercase
from get_language_dataset import get_language_dataset
from get_default_params import get_params
from udify import util
from ourmaml import MAML, maml_update
from udify.predictors import predictor
from allennlp.common.util import prepare_environment
from allennlp.models.model import Model
from allennlp.models.archival import archive_model
import allennlp
from schedulers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch import autograd
from torch.optim import Adam
import torch
import numpy as np
import argparse
import subprocess
import json
import sys
import os

from allennlp.nn.util import move_to_device
sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip_update", default=0, type=float, help="Skip update on the support set")
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument("--support_set_size", default=1, type=int, help="Support set size")
    parser.add_argument("--maml", default=False, type=bool, help="Do MAML instead of XMAML, that is, include English as an auxiliary task if flag is set and start from scratch",)
    parser.add_argument( "--addenglish", default=False, type=bool, help="Add English as a task" )
    parser.add_argument( "--notaddhindi", default=False, type=bool, help="Add English as a task" ) #?
    parser.add_argument("--episodes", default=1, type=int, help="Amount of episodes")
    parser.add_argument( "--updates", default=5, type=int, help="Amount of inner loop updates" )
    parser.add_argument("--name", default="", type=str, help="Name to add")
    parser.add_argument( "--meta_lr_decoder", default=0.001, type=float, help="Meta adaptation LR for the decoder")
    parser.add_argument( "--meta_lr_bert", default=0.001, type=float, help="Meta adaptation LR for BERT" )
    parser.add_argument( "--inner_lr_decoder", default=0.001, type=float, help="Inner learner LR for the decoder" )
    parser.add_argument( "--inner_lr_bert", default=0.001, type=float, help="Inner learner LR for BERT" )
    parser.add_argument( "--model_dir", default=None, type=str, help="Directory from where to start training. Should be a 'clean' model for MAML and a pretrained model for X-MAML.",    )
    args = parser.parse_args()

    from pathlib import Path
    Path("saved_models").mkdir(parents=True, exist_ok=True)

    training_tasks = []
    torch.cuda.empty_cache()
    # 7 languages by default -R
    for lan, lan_l in zip(train_languages, train_languages_lowercase):
        #print(f"[INFO]: Creating dataset for language {lan}")
        if "indi" in lan and not args.notaddhindi:
            training_tasks.append(
                get_language_dataset(
                    lan, lan_l, seed=args.seed, support_set_size=args.support_set_size
                )
            )
        elif "indi" in lan and args.notaddhindi:
            continue
        elif "indi" not in lan and not args.notaddhindi: #this gets called as default -R
            training_tasks.append(
                get_language_dataset(
                    lan, lan_l, seed=args.seed, support_set_size=args.support_set_size
                )
            )
        elif "indi" not in lan and args.notaddhindi:
            training_tasks.append(
                get_language_dataset(
                    lan, lan_l, seed=args.seed, support_set_size=args.support_set_size
                )
            )
            
    # Setting parameters
    DOING_MAML = args.maml
    if DOING_MAML or args.addenglish:
        # Get another training task
        training_tasks.append(
            get_language_dataset(
                "UD_English-EWT",
                "en_ewt-ud",
                seed=args.seed,
                support_set_size=args.support_set_size,
            )
        )
    UPDATES = args.updates
    EPISODES = args.episodes
    INNER_LR_DECODER = args.inner_lr_decoder
    INNER_LR_BERT = args.inner_lr_bert
    META_LR_DECODER = args.meta_lr_decoder
    META_LR_BERT = args.meta_lr_bert
    SKIP_UPDATE = args.skip_update

    # Filenames
    MODEL_FILE = ( 
        args.model_dir
        if args.model_dir is not None
        else ("logs/bert_finetune_en/2021.05.13_01.56.30"
            # "../backup/pretrained/english_expmix_deps_seed2/2020.07.30_18.50.07"
            # if not DOING_MAML
            # else "logs/english_expmix_tiny_deps2/2020.05.29_17.59.31"
        )
    )
    train_params = get_params("metalearning", args.seed)

    m = Model.load(
        train_params,
        MODEL_FILE,
    )

    maml_string = "saved_models/MAML" if DOING_MAML else "saved_models/XMAML"
    param_list = [
        str(z)
        for z in [
            maml_string,
            INNER_LR_DECODER,
            INNER_LR_BERT,
            META_LR_DECODER,
            META_LR_BERT,
            UPDATES,
            args.seed,
        ]
    ]
    MODEL_SAVE_NAME = "_".join(param_list)
    MODEL_VAL_DIR = MODEL_SAVE_NAME + args.name
    META_WRITER = MODEL_VAL_DIR + "/meta_results.txt"

    if not os.path.exists(MODEL_VAL_DIR):
        subprocess.run(["mkdir", MODEL_VAL_DIR])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
        subprocess.run(["cp", "-r", MODEL_FILE + "/vocabulary", MODEL_VAL_DIR])
        subprocess.run(["cp", MODEL_FILE + "/config.json", MODEL_VAL_DIR])

    with open(META_WRITER, "w") as f:
        f.write("Model ready\n")

    # Loading the model
    train_params = get_params("metalearning", args.seed)
    prepare_environment(train_params)
    m = Model.load(
        train_params,
        MODEL_FILE,
    )
    meta_m = MAML(
        m, INNER_LR_DECODER, INNER_LR_BERT, first_order=True, allow_unused=True
    ).cuda()
    optimizer = Adam(
        [
            {
                "params": meta_m.module.text_field_embedder.parameters(),
                "lr": META_LR_BERT,
            },
            {"params": meta_m.module.decoders.parameters(), "lr": META_LR_DECODER},
            {"params": meta_m.module.scalar_mix.parameters(), "lr": META_LR_DECODER},
        ],
        META_LR_DECODER,
    )  # , weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 50, 500)

    # NI START
    print(f"[INFO]: Total amount of training tasks is {len(training_tasks)}")
    gradients_for_ni = torch.Tensor()

    gradients_for_ni = [] # in the end it will store the following info in each dim
    # num_episodes x grad_len x num_languages

    # NI END


    for iteration in range(EPISODES):
        print(f"[INFO]: Starting episode {iteration}", flush=True)
        iteration_loss = 0.0
        # NI START
        episode_grads = []
        # NI END
        """Inner adaptation loop"""
        for j, task_generator in enumerate(training_tasks):

            # NI start
            language_grads = torch.Tensor()
            #print(f"[INFO:] Meta-training language", train_languages[j])
            #gradients_for_ni = torch.Tensor()

            #print(f"[INFO]: Training taksk has length {len(task_generator)}")
            # NI end

            learner = meta_m.clone()

            # Sample two batches
            support_set = next(iter(task_generator))
            support_set = move_to_device(support_set,torch.device('cuda'))
            if SKIP_UPDATE == 0.0 or torch.rand(1) > SKIP_UPDATE:
                for mini_epoch in range(UPDATES):
                    # print(support_set)

                    torch.cuda.empty_cache()
                    inner_loss = learner.forward(**support_set)["loss"]
                    # NI START
                    
                    # learner.adapt(inner_loss, first_order=True)
                    # The following two lines  implemnt learning.adapt. See our_maml.py for details
                    grads = autograd.grad(inner_loss, learner.parameters(), create_graph=False, allow_unused=True)
                    maml_update(learner, lr=args.inner_lr_decoder, lr_small=args.inner_lr_bert, grads=grads)        

                    # print("SHAPES")
                    # new_grads = []# filters out None grads
                    # for i in grads:
                    #     print(type(i))
                    #     if type(i) == torch.Tensor:
                    #         #print(i.shape)
                    #         new_grads.append(i)
                        

                    #print("torch stack",  torch.stack(new_grads).shape)
                    #print("torch stack",  torch.stack(grads).shape)

                    # grads_to_save = torch.stack(grads[0]).reshape(-1)
                    grads_to_save = grads[0].reshape(-1) # grads[0] are mBERT parameters (?)
                    #print(grads[0].shape)
                    #print(len(grads))
                    #print(grads[1].shape)

                    language_grads = torch.cat([language_grads.cpu(), grads_to_save.cpu()], dim=-1) # Updates*grad_len in the last update

                    #print(gradients_for_ni.shape)
                    # NI end

                    del inner_loss
                    torch.cuda.empty_cache()
            
            # NI start
            language_grads = language_grads.reshape(-1, UPDATES) # setup for taking the average
            language_grads = torch.mean(language_grads, dim = 1) # number of gradients x 1
            #print("Language grads shape", language_grads.shape)
            episode_grads.append(language_grads.detach().numpy())
            #torch.save()
            # NI end

            del support_set
            query_set = next(iter(task_generator))
            query_set = move_to_device(query_set,torch.device('cuda'))

            eval_loss = learner.forward(**query_set)["loss"]
            iteration_loss += eval_loss

            del eval_loss
            del learner
            del query_set
            torch.cuda.empty_cache()

        ### NI START
        gradients_for_ni.append(np.array(episode_grads))
        #### NI end

        # Sum up and normalize over all 7 losses
        iteration_loss /= len(training_tasks)
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()
        scheduler.step()

        # Bookkeeping
        with torch.no_grad():
            print(iteration, "meta", iteration_loss.item())
            with open(META_WRITER, "a") as f:
                f.write(str(iteration) + " meta " + str(iteration_loss.item()))
                f.write("\n")
        del iteration_loss
        torch.cuda.empty_cache()

        if iteration + 1 in [1,500, 1500, 2000] and not (
            iteration + 1 == 500 and DOING_MAML
        ):
            backup_path = os.path.join(
                MODEL_VAL_DIR, "model" + str(iteration + 1) + ".th"
            )
            torch.save(meta_m.module.state_dict(), backup_path)

    ### NI START
    save_this = torch.from_numpy(np.array(gradients_for_ni))
    print(f"[INFO]: Saving the gradients with shape {save_this.shape}")
    torch.save(save_this, f"gradients_for_ni_epi{EPISODES}_upd{UPDATES}_suppSize{args.support_set_size}")

    ### NI END
    print("Done training ... archiving three models!")
    for i in [1,500, 600, 900, 1200, 1500, 1800, 2000, 1500]:
        filename = os.path.join(MODEL_VAL_DIR, "model" + str(i) + ".th")
        print(filename)
        if os.path.exists(filename):
            print('exists')
            save_place = MODEL_VAL_DIR + "/" + str(i)
            subprocess.run(["mv", filename, MODEL_VAL_DIR + "/best.th"])
            subprocess.run(["mkdir", save_place])
            archive_model(
                MODEL_VAL_DIR,
                # files_to_archive=train_params.files_to_archive,
                archive_path=save_place,
            )
    subprocess.run(["rm", MODEL_VAL_DIR + "/best.th"])


if __name__ == "__main__":
    main()
