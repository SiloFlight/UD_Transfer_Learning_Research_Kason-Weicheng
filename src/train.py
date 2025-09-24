import sys
import typing

from transformers import AutoTokenizer,DataCollatorForTokenClassification,TrainingArguments,Trainer,AutoModelForTokenClassification
import numpy as np
from sklearn.metrics import f1_score

import dataset_helper

program_call_format = "train.py training_code epochs max_training_size"
example_program_call = "train.py en_ewt 5 -1"

## Configs
BATCH_SIZE = 16

def validateProgramCall():
    if len(sys.argv) != len(program_call_format.split(" ")):
        print("Invalid program call length")
        return False
    
    _, training_code, epochs, max_training_size = sys.argv
    #Validate training_code
    if training_code not in typing.get_args(dataset_helper.DatasetCode):
        print(f"Invalid training code provided: {training_code}")
        return False

    #Validate epochs
    try:
        int(epochs)
    except:
        print(f"Epochs is not a number.")
        return False
    
    #Validate max_training_size
    try:
        int(max_training_size)
    except:
        print(f"Max_training_size is not a number.")
        return False
    
    return True

def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=2)

    y_true, y_pred = [], []
    for pred_row, lab_row in zip(preds, labels):
        for p_i, l_i in zip(pred_row, lab_row):
            if l_i != -100:         # ignore subword/CLS/SEP
                y_true.append(l_i)
                y_pred.append(p_i)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    f1  = f1_score(y_true, y_pred, average="macro")  # macro over 17 UPOS classes
    return {"token_acc": acc, "f1_macro": f1}


from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification

UPOS_TAGS = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

label2id = {l:i for i,l in enumerate(UPOS_TAGS)}
id2label = {i:l for l,i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def get_pre_tuned_model():
    return AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased",
        num_labels=len(UPOS_TAGS),
        label2id=label2id,
        id2label=id2label,
    )

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def train_model(model,output_directory,training_set,eval_set,epochs):
    from transformers import TrainingArguments,Trainer

    training_args = TrainingArguments(
        output_dir=output_directory,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=eval_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer

def tokenize_and_align(unprocessedTokens):
    tokenized = tokenizer(unprocessedTokens["tokens"],is_split_into_words=True,truncation=True)

    labels = []
    for i,label in enumerate(unprocessedTokens["upos"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx == None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def main():
    if not validateProgramCall():
        return
    
    _, training_code, epochs, max_training_size = sys.argv

    epochs = int(epochs)
    max_training_size = int(max_training_size)
    training_code = typing.cast(dataset_helper.DatasetCode,training_code)

    
    training_set = dataset_helper.get_dataset(training_code,"train")
    eval_set = dataset_helper.get_dataset(training_code,"test")

    cols_to_remove = training_set.column_names

    if max_training_size != -1:
        training_set = training_set.shuffle(seed=42).select(range(min(len(training_set),max_training_size)))

    training_set = training_set.map(tokenize_and_align,batched=True,remove_columns=cols_to_remove)
    eval_set = eval_set.map(tokenize_and_align,batched=True,remove_columns=cols_to_remove)

    model = get_pre_tuned_model()

    trained = train_model(model,f"models/{training_code}_model",training_set,eval_set,epochs)

    
    
    

if __name__ == "__main__":
    main()