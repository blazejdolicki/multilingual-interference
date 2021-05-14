"""
Split test files for language that only have a test set, such as Swedish, Faroese and Breton.
"""
import argparse
import os
import random
from typing import Dict, Tuple, List, Any, Callable
from naming_conventions import *

def lazy_nonparse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [line for line in sentence.split("\n")]


parser = argparse.ArgumentParser()
# parser.add_argument("--file", default="data/ud-treebanks-v2.3/UD_Breton-KEB/br_keb-ud-test.conllu", type=str, help="File to split up")
parser.add_argument("--amt", default=20, type=int, help="How many pieces to split up")
parser.add_argument("--batch_size", default=20, type=int, help="How big is each batch")
parser.add_argument("--seed", default=2002, type=int, help="Seed for the shuffler")

args = parser.parse_args()
from pathlib import Path
Path("data/ud-tiny-treebanks/size" + str(args.batch_size)).mkdir(parents=True, exist_ok=True)


# if args.file == None:
#     with open("all_mini_sets.txt", "r", encoding="utf-8") as conllu_file:
small = languages_too_small_for_20_batch_20
lower = languages_too_small_for_20_batch_20_lowercase
test_files = ["data/ud-treebanks-v2.3/"+small[i]+"/"+lower[i]+"-test.conllu"for i in range(len(small))]

for file in test_files:

    output_filename = os.path.join(
        "data/ud-tiny-treebanks/size" + str(args.batch_size),
        file.strip("-test.conllu").split("/")[-1],
    )

    annotations = []
    with open(file, "r", encoding="utf-8") as conllu_file:
        for annotation in lazy_nonparse(conllu_file.read()):
            annotations.append(annotation)

    for i in range(args.amt):
        random.shuffle(annotations)
        development = annotations[: args.batch_size]
        test = annotations[args.batch_size :]

        # new Dev
        with open(output_filename + "-dev" + str(i) + ".conllu", "w") as f:
            for z in development:
                for line in z:
                    f.write(line)
                    f.write("\n")
                f.write("\n")
        # new Test
        with open(output_filename + "-test" + str(i) + ".conllu", "w") as f:
            for z in test:
                for line in z:
                    f.write(line)
                    f.write("\n")
                f.write("\n")
