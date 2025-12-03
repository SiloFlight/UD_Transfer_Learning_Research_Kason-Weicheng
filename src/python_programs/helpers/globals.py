from typing import Literal

from os import path

project_path = path.join(path.dirname(path.abspath(__file__)),"../../..")

language_codes = ["en","es","fr","fa","pt","ur","ug","vi","de","zh","he","ar"]

dataset_types = ["train","test","dev"]

model_types = ["mbert","xlmr"]

lang2directory = {
    "en" : "UD_English-EWT",
    "es" : "UD_Spanish-GSD",
    "fr" : "UD_French-GSD",
    "fa" : "UD_Persian-PerDT",
    "pt" : "UD_Portuguese-GSD",
    "ur" : "UD_Urdu-UDTB",
    "ug" : "UD_Uyghur-UDT",
    "vi" : "UD_Vietnamese-VTB",
    "de" : "UD_German-GSD",
    "zh" : "UD_Chinese-GSD",
    "he" : "UD_Hebrew-HTB",
    "ar" : "UD_Arabic-PADT",
}

lang2UDcode = {
    "en" : "en_ewt",
    "es" : "es_gsd",
    "fr" : "fr_gsd",
    "fa" : "fa_perdt",
    "pt" : "pt_gsd",
    "ur" : "ur_udtb",
    "ug" : "ug_udt",
    "vi" : "vi_vtb",
    "de" : "de_gsd",
    "zh" : "zh_gsd",
    "he" : "he_htb",
    "ar" : "ar_padt",
}

model2modelname = {
    "mbert" : "bert-base-multilingual-cased",
    "xlmr" : "xlm-roberta-base",
}

UPOS_TAGS = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]
label2id = {l:i for i,l in enumerate(UPOS_TAGS)}
id2label = {i:l for l,i in label2id.items()}