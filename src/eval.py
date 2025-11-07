import sys
import typing
import os

from transformers import AutoTokenizer,DataCollatorForTokenClassification,TrainingArguments,Trainer,AutoModelForTokenClassification
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score

import dataset_helper,model_helper

program_call_format = "eval.py model_code eval_code"
example_program_call = "eval.py en_ewt en_ewt"

## Configs
BATCH_SIZE = 16



from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification

UPOS_TAGS = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

label2id = {l:i for i,l in enumerate(UPOS_TAGS)}
id2label = {i:l for l,i in label2id.items()}

label_list = [id2label[i] for i in range(len(id2label))]

def validateProgramCall():
    if len(sys.argv) != len(program_call_format.split(" ")):
        print("Invalid program call length")
        return False
    
    _, model_code, eval_code = sys.argv
    #Validate model_code
    if model_code not in typing.get_args(dataset_helper.DatasetCode):
        print(f"Invalid training code provided: {model_code}")
        return False
    
    #Validate eval_code
    if eval_code not in typing.get_args(dataset_helper.DatasetCode):
        print(f"Invalid training code provided: {eval_code}")
        return False
    
    return True

label_ids = np.arange(len(label_list))

def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=2)

        y_true, y_pred = [], []
        for pred_row, lab_row in zip(preds, labels):
            for p_i, l_i in zip(pred_row, lab_row):
                if l_i != -100:
                    y_true.append(l_i)
                    y_pred.append(p_i)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Overall metrics
        acc = (y_true == y_pred).mean()
        f1_macro = f1_score(y_true, y_pred, average="macro")

        # Per-tag metrics (aligned to label_list / label_ids)
        prec, rec, f1, supp = precision_recall_fscore_support(
            y_true, y_pred,
            labels=label_ids,
            average=None,
            zero_division=0
        )

        assert isinstance(prec, np.ndarray)
        assert isinstance(rec, np.ndarray)
        assert isinstance(f1, np.ndarray)
        assert isinstance(supp, np.ndarray)

        # Flatten into a dict of scalars for HF Trainer
        metrics = {
            "token_acc": acc,
            "f1_macro": f1_macro,
        }
        for i, tag in enumerate(label_list):
            metrics[f"prec_{tag}"]        = float(prec[i])
            metrics[f"f1_{tag}"]        = float(f1[i])
            metrics[f"support_{tag}"] = float(supp[i])

        return metrics

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

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
    
    _, model_code, eval_code = sys.argv

    model_code = typing.cast(dataset_helper.DatasetCode,model_code)
    eval_code = typing.cast(dataset_helper.DatasetCode,eval_code)

    model = model_helper.get_model(model_code)
    eval_set = dataset_helper.get_dataset(eval_code,"test")

    trainer = Trainer(
        model,
        eval_dataset=eval_set.map(tokenize_and_align,batched=True,remove_columns=eval_set.column_names),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    res = trainer.evaluate()

    save_path = f"results/{model_code}-{eval_code}.txt"
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    with open(save_path,"w") as f:
        f.write(str(res))

    
    
    

if __name__ == "__main__":
    main()