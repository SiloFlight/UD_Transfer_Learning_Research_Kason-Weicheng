import sys,numpy as np,pandas as pd,os
from os import path
import json
import lang2vec.lang2vec as l2v
from transformers import AutoTokenizer
from scipy.stats import spearmanr

from .helpers.globals import *
from .helpers import datasets

def loadEvalData(model_type,langs):
    data = {}

    for trained_lang in langs:
        data[trained_lang] = {}
        
        for eval_lang in langs:
            file_name = path.join(project_path,f"results/{model_type}_{trained_lang}_{eval_lang}.json")

            with open(file_name,"r") as f:
                data[trained_lang][eval_lang] = json.load(f)
    
    return data

def getSubwordSet(model_type,lang):
    dataset = datasets.get_dataset(lang,"train")
    tokenizer = AutoTokenizer.from_pretrained(model2modelname[model_type])

    subword_vocab = set()

    for i in range(len(dataset)):
        tokens = dataset[i]["tokens"]

        text = " ".join(tokens)
        wordpieces = tokenizer.tokenize(text)
        subword_vocab.update(wordpieces)

    return subword_vocab


def main():
    _,model_type = sys.argv[:2]
    langs = sys.argv[2:]

    eval_data = loadEvalData(model_type,langs)

    ## Create Overlap Matrix
    lang_subwords = {lang:getSubwordSet(model_type,lang) for lang in langs}

    overlap_matrix = pd.DataFrame(index=langs, columns=langs, dtype=float)

    for a in langs:
        denom = len(lang_subwords[a]) if len(lang_subwords[a]) > 0 else 1
        for b in langs:
            inter = lang_subwords[a].intersection(lang_subwords[b])

            rate = len(inter)/denom

            overlap_matrix.loc[a,b] = rate


    ## Create Lang distance Matrix
    distance_matrix = pd.DataFrame(index=langs, columns=langs, dtype=float)
    for a in langs:
        for b in langs:
            d = l2v.distance("genetic",a, b)
            distance_matrix.loc[a,b] = d

    ## Create Accuracy Matrix
    records = []
    for trained_lang in langs:
        for eval_lang in langs:
            records.append({
            "train_lang": trained_lang,
            "eval_lang": eval_lang,
            "accuracy": eval_data[trained_lang][eval_lang]["token_acc"]
        })
    df = pd.DataFrame(records)
    
    accuracy_matrix = df.pivot(index="train_lang", columns="eval_lang", values="accuracy")

    spearman_results = {}

    for trained_lang in accuracy_matrix.index:   
        accuracies = accuracy_matrix.loc[trained_lang]
        distances = distance_matrix.loc[trained_lang]

        rho, pval = spearmanr(distances, accuracies)
        spearman_results[trained_lang] = {"rho": rho, "p_value": pval}

    distance_spearman = pd.DataFrame(spearman_results).T

    results = []
    for target in accuracy_matrix.index:
        acc = accuracy_matrix.loc[target]
        ov  = overlap_matrix.loc[target]

        rho, p = spearmanr(ov, acc)
        results.append({"trained_lang": target, "rho": rho, "p_value": p})

    overlap_spearman = pd.DataFrame(results)

    results_dir = path.join(project_path, "results", model_type)
    os.makedirs(results_dir, exist_ok=True)
    overlap_matrix.to_csv(path.join(results_dir, f"{model_type}_{"-".join(langs)}_overlap.csv"))
    distance_matrix.to_csv(path.join(results_dir, f"{model_type}_{"-".join(langs)}_distance.csv"))
    accuracy_matrix.to_csv(path.join(results_dir, f"{model_type}_{"-".join(langs)}_accuracy.csv"))
    distance_spearman.to_csv(path.join(results_dir, f"{model_type}_{"-".join(langs)}_spearman_distance.csv"))
    overlap_spearman.to_csv(path.join(results_dir, f"{model_type}_{"-".join(langs)}_spearman_overlap.csv"))






if __name__ == "__main__":
    main()
