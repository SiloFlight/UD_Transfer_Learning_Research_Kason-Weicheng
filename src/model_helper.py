from os import path
import os
from transformers import AutoModelForTokenClassification

from my_globals import *

def get_rel_model_path(model_code : DatasetCode) -> str:
    model_dir = f"../models/{model_code}_model"
    res_dir = f"models/{model_code}_model"

    dir_path = path.join(os.path.dirname(os.path.realpath(__file__)),model_dir)

    curr_iter = 0
    curr_dir = ""

    for entry_name in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry_name)
        if os.path.isdir(full_path):
            x = int(entry_name.split("-")[-1])
            if x > curr_iter:
                curr_dir = res_dir+"/"+entry_name
        
    return curr_dir

def get_model(model_code : DatasetCode):
    return AutoModelForTokenClassification.from_pretrained(get_rel_model_path(model_code),
        num_labels=len(UPOS_TAGS),
        label2id=label2id,
        id2label=id2label,
        
    )