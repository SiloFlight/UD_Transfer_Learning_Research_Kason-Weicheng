from datasets import Dataset
from conllu import parse_incr
from os import path

from .globals import lang2UDcode,lang2directory,project_path

def readConllu(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            tokens, upos = [], []
            for tok in sent:
                # skip multi-word tokens whose 'id' is a range like "3-4"
                if isinstance(tok["id"], int):
                    tokens.append(tok["form"])
                    upos.append(tok["upostag"])  # CoNLL-U "UPOS" field
            rows.append({"tokens": tokens, "upos": upos})
    return rows

def get_dataset(lang_code,dataset_type):
    dataset_directory = path.join(project_path,"datasets",lang2directory[lang_code])
    UD_code = lang2UDcode[lang_code]

    dataset_path = path.join(dataset_directory,f"{UD_code}-ud-{dataset_type}.conllu")

    return Dataset.from_list(readConllu(dataset_path))
