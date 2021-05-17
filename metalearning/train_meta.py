# -*- coding: utf-8 -*-
"""
This file Meta-Trains on 7 languages
And validates on Bulgarian
"""
import naming_conventions
from get_language_dataset import get_language_dataset
from get_default_params import get_params
from ourmaml import MAML, maml_update
from allennlp.common.util import prepare_environment
from allennlp.models.model import Model
from allennlp.models.archival import archive_model
from schedulers import get_cosine_schedule_with_warmup
from torch import autograd
from torch.optim import Adam
import torch
import numpy as np
import argparse
import subprocess
import sys
import os, glob


from allennlp.nn.util import move_to_device
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

torch.cuda.empty_cache()  # please stop oom's


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip_update", default=0, type=float, help="Skip update on the support set")
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument("--support_set_size", default=1, type=int, help="Support set size")
    parser.add_argument("--maml", default=False, type=bool, help="Do MAML instead of XMAML, that is, include English as an auxiliary task if flag is set and start from scratch")
    parser.add_argument("--addenglish", default=False, type=bool, help="Add English as a task")
    parser.add_argument("--notaddhindi", default=False, type=bool, help="Add English as a task") #?
    parser.add_argument("--episodes", default=1, type=int, help="Amount of episodes")
    parser.add_argument("--updates", default=5, type=int, help="Amount of inner loop updates")
    parser.add_argument("--name", default="", type=str, help="Name to add")
    parser.add_argument("--meta_lr_decoder", default=0.001, type=float, help="Meta adaptation LR for the decoder")
    parser.add_argument("--meta_lr_bert", default=0.001, type=float, help="Meta adaptation LR for BERT")
    parser.add_argument("--inner_lr_decoder", default=0.001, type=float, help="Inner learner LR for the decoder")
    parser.add_argument("--inner_lr_bert", default=0.001, type=float, help="Inner learner LR for BERT")
    parser.add_argument("--model_dir", default=None, type=str, help="Directory from where to start training. Should be a 'clean' model for MAML and a pretrained model for X-MAML.")
    parser.add_argument("--language_order", default=0, type=int, help="The order of languages in the inner loop")
    parser.add_argument("--save_every", default=10, type=int, help="Save the gradient conflicts every save_every episodes ")
    parser.add_argument("--accumulation_mode", default="sum", type=str, help="What gradient accumulation strategy to use", choices=["mean", "sum"])
    args = parser.parse_args()

    from pathlib import Path
    Path("saved_models").mkdir(parents=True, exist_ok=True)

    print(f"Using accumulation mode {args.accumulation_mode} for gradient accumulation")
    device = torch.device('cuda')

    training_tasks = []
    torch.cuda.empty_cache()

    # 7 languages by default -R
    if args.language_order == 0:
        lan_ = naming_conventions.train_languages
        lan_lowercase_ = naming_conventions.train_languages_lowercase

    elif args.language_order == 1:
        lan_ = naming_conventions.train_languages_order_1
        lan_lowercase_ = naming_conventions.train_languages_order_1_lowercase

    elif args.language_order == 2:
        lan_ = naming_conventions.train_languages_order_2
        lan_lowercase_ = naming_conventions.train_languages_order_2_lowercase

    for lan, lan_l in zip(lan_, lan_lowercase_):
        if not ("indi" in lan and args.notaddhindi):
            training_tasks.append(get_language_dataset(lan, lan_l, seed=args.seed, support_set_size=args.support_set_size))
            
    # Setting parameters
    DOING_MAML = args.maml
    if DOING_MAML or args.addenglish:
        training_tasks.append(get_language_dataset( "UD_English-EWT", "en_ewt-ud", seed=args.seed, support_set_size=args.support_set_size))

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
            args.language_order,
            args.accumulation_mode
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
    # gradients_for_ni = torch.Tensor()

    # gradients_for_ni = [] # in the end it will store the following info in each dim
    cos_matrices = []
    # num_episodes x grad_len x num_languages

    # NI END

    def restart_iter(task_generator,args): 
        """ Restart the iter(Dataloader) by creating again the dataset.
        This method is called when we looped through the whole data and want to start from the beginning.
        """       
        # Get the path to datat like: data/ud-treebanks-v2.3/UD_Arabic-PADT/ar_padt-ud-train.conllu
        task_split = task_generator._dataset._file_path.split('/')
        lan = task_split[-2] # Language in capitals
        # Language in lower case, remove -train.collu, -dev.conllu, -test.conllu
        lan_lowercase_ = task_split[-1].split('-')[0]+'-ud'
        return get_language_dataset(lan, lan_lowercase_, seed=args.seed, support_set_size=args.support_set_size)
    
           
    for iteration in range(EPISODES):
        print(f"[INFO]: Starting episode {iteration}", flush=True)
        iteration_loss = 0.0
        ### NI START
        episode_grads = [] # store the gradients of an episode for all languages
        ### NI END
        """Inner adaptation loop"""
        for j, task_generator in enumerate(training_tasks):

            ### NI start
            language_grads = torch.Tensor()#.to(device=device)
            ### NI end

            learner = meta_m.clone()

            # Sample two batches
            try:
                support_set = next(task_generator) 
            except StopIteration: #Exception called if iter reached its end.
                #We create a new iterator to use instead
                training_tasks[j] = restart_iter(task_generator,args)                
                task_generator =training_tasks[j] 
                support_set = next(task_generator) #Sample from new iter

            support_set = move_to_device(support_set,torch.device('cuda'))
            if SKIP_UPDATE == 0.0 or torch.rand(1) > SKIP_UPDATE:
                for mini_epoch in range(UPDATES):
                    # print(support_set)

                    torch.cuda.empty_cache()
                    inner_loss = learner.forward(**support_set)["loss"]
                    ### NI START
                    
                    # learner.adapt(inner_loss, first_order=True)
                    # The following two lines  implemnt learning.adapt. See our_maml.py for details
                    grads = autograd.grad(inner_loss, learner.parameters(), create_graph=False, allow_unused=True)
                    maml_update(learner, lr=args.inner_lr_decoder, lr_small=args.inner_lr_bert, grads=grads)        


                    #print(gradients_for_ni.shape)
                    ### NI end

                    del inner_loss
                    torch.cuda.empty_cache()

                ### NI start
                if (iteration+1)%args.save_every==0:
                        new_grads = []# filters out None grads
                        for i in grads:
                            #print(type(i))
                            if type(i) == torch.Tensor:
                                #print(i.shape)
                                #new_grads.append(i.detach().reshape(-1))
                                new_grads.append(i.reshape(-1))
                
                        #grads_to_save = grads[0].detach().reshape(-1) # grads[0] are mBERT parameters (?)
                        grads_to_save = torch.hstack(new_grads) # getting all the parameters
                        #print(f"Shape of grads to save", grads_to_save.shape)

                        #language_grads = torch.cat([language_grads.cpu(), grads_to_save.cpu()], dim=-1) # Updates*grad_len in the last update
                        language_grads = torch.cat([language_grads.cpu(), grads_to_save.cpu()], dim=-1) # Updates*grad_len in the last update
            
                ### NI end

            ### NI start
            if (iteration+1)%args.save_every==0:
                language_grads = language_grads.reshape(-1) # setup for taking the average

                # if args.accumulation_mode=="mean":
                #     language_grads = torch.mean(language_grads, dim = 1) # number of gradients x 1
                # else: # args.accumulation_mode=="sum"
                #     language_grads = torch.sum(language_grads, dim = 1) # number of gradients x 1
                
                #episode_grads.append(language_grads.detach().numpy())
                episode_grads.append(language_grads)

            ### NI end

            del support_set 
            try:
                query_set = next(task_generator)
            except StopIteration: #Exception called if iter reached its end.
                #We create a new iterator to use instead
                training_tasks[j] = restart_iter(task_generator,args)
                task_generator =training_tasks[j] 
                query_set = next(task_generator)
            query_set = move_to_device(query_set,torch.device('cuda'))

            eval_loss = learner.forward(**query_set)["loss"]
            iteration_loss += eval_loss

            del eval_loss
            del learner
            del query_set
            torch.cuda.empty_cache()

        ### NI START
        #gradients_for_ni.append(np.array(episode_grads)) 

        #Does saving epiosde grads seperatetly remedy the issue? yes!
        # if (iteration+1)%10==0: 
            # save_this = torch.from_numpy(np.array(episode_grads))
            # print(f"[INFO]: Saving the gradients with shape {save_this.shape}")
            # torch.save(save_this, f"saved_grads/epiosde_grad{iteration}_upd{UPDATES}_suppSize{args.support_set_size}")

        #print(f"[INFO]: Saving the similarity matrix with shape {cos_matrix.shape}")
        #np.save(f"saved_grads/epiosde_grad{iteration}_upd{UPDATES}_suppSize{args.support_set_size}")
            
        #### NI end
        #grad_copy = torch.stack((episode_grads)).detach() # detaching the grad
        # Sum up and normalize over all 7 losses
        iteration_loss /= len(training_tasks)
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()
        scheduler.step()

        if (iteration+1)%args.save_every==0:
            #epi_grads = np.array(episode_grads)
            #for language_grad in episode_grads:
            #    language_grads_detachted
            #    for grad in language_grads:

           # epi_grads = torch.Tensor(episode_grads)#.detach()
            epi_grads = []

            # for epi_grad in episode_grads:
            #     lan_grads_detachted = []
            #     print(f"epi grad shape {epi_grad.shape}")
            #     for lan_grad in epi_grad:
            #         #print(f"language gradient shape{lan_grad.shape}")
            #         lan_grads_detachted.append(lan_grad.detach())
                # epi_grad.append(lan_grads_detachted)

            epi_grads = torch.stack((episode_grads))#.detach()
            #print(f"Is detachted vector different than one that's not {torch.equal(epi_grads, grad_copy)}")
            print("[INFO]: Calculating cosine similarity matrix ...")
            cos_matrix = cosine_similarity(epi_grads)
            #print("Cos sim matrix shape", cos_matrix.shape)
            cos_matrices.append(np.array(cos_matrix)) 
            print("Cos matrices shape", np.array(cos_matrices).shape)

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
        # Save the gradients in case OOM occurs
        # Can't escape OOM

        if (iteration+1)%args.save_every==0:# not to slow down a lot
            
            save_this = np.array(cos_matrices)

            # Delete the last temp file
            for filename in glob.glob(f"./cos_matrices/temp_allGrads_episode_upd{UPDATES}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}_cos_mat{iteration}*"): # remove the previoustemp grads
                os.remove(filename) 

            np.save(f"cos_matrices/temp_allGrads_episode_upd{UPDATES}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}_cos_mat{iteration}",
                    save_this)
            torch.cuda.empty_cache()

        ### NI END


    ### NI START
    # save_this = torch.from_numpy(np.array(gradients_for_ni))
    # print(f"[INFO]: Saving the gradients with shape {save_this.shape}")
    # torch.save(save_this, f"saved_grads/gradients_for_ni_epi{EPISODES}_upd{UPDATES}_suppSize{args.support_set_size}")

    # Delete the last temp file
    for filename in glob.glob(f"./cos_matrices/temp_allGrads_episode_upd{UPDATES}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}*"): # remove the previoustemp grads
        os.remove(filename) 
    cos_matrices = np.array(cos_matrices)
    print(f"[INFO]: Saving the similarity matrix with shape {cos_matrices.shape}")
    np.save(f"cos_matrices/allGrads_episode_upd{UPDATES}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}_cos_mat{EPISODES}", cos_matrices)

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
