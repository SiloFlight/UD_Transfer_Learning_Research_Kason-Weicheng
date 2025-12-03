from transformers import AutoModelForTokenClassification
from os import path
import os

from .globals import project_path,UPOS_TAGS,label2id,id2label

def get_trained_model(lang_code, model_type):
    output_dir = path.join(project_path,f"models/{lang_code}_{model_type}")
    if not path.isdir(output_dir):
        return None

    max_index = -1

    model_dir = ""

    for dir_name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, dir_name)

        if not dir_name.startswith("checkpoint-"):
            continue

        try:
            model_index = int(dir_name.split("-")[-1])
        except Exception:
            continue

        if model_index > max_index:
            max_index = model_index

            model_dir = full_path
    
    if model_dir == "":
        return None
    else:
        return AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=len(UPOS_TAGS),
            label2id=label2id,
            id2label=id2label,
        )
