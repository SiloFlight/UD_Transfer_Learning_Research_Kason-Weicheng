from typing import Literal

from datasets import Dataset
from conllu import parse_incr
from os import path

from src.my_globals import *

datasetCode2Directory : dict[DatasetCode,str] = {
    "en_ewt" : "../datasets/UD_English-EWT",
    "fr_gsd" : "../datasets/UD_French-GSD",
    "es_gsd" : "../datasets/UD_Spanish-GSD",
    "pt_gsd" : "../datasets/UD_Portuguese-GSD",
    "ur_udtb" : "../datasets/UD_Urdu-UDTB" ,
    "ug_udt" : "../datasets/UD_Uyghyr-UDT",
    "vi_vtb" : "../datasets/UD_Vietnamese-VTB",
    "fa_perdt" : "../datasets/UD_Persian-PerDT"
}

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

def get_dataset(dataset_code : DatasetCode, dataset_type : DatasetType) -> Dataset:
    file_dir = path.dirname(path.abspath(__file__))
    dataset_path = path.join(file_dir,datasetCode2Directory[dataset_code],f"{dataset_code}-ud-{dataset_type}.conllu")

    return Dataset.from_list(readConllu(dataset_path))