from typing import Literal

DatasetCode = Literal["en_ewt","fr_gsd","es_gsd","pt_gsd","ur_udtb","ug_udt","vi_vtb","fa_perdt"]
DatasetType = Literal["train","test","dev"]

UPOS_TAGS = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]
label2id = {l:i for i,l in enumerate(UPOS_TAGS)}
id2label = {i:l for l,i in label2id.items()}

