import sys,numpy as np
from os import path
from sklearn.metrics import f1_score
from transformers import AutoModelForTokenClassification,AutoTokenizer,DataCollatorForTokenClassification,TrainingArguments,Trainer


from .helpers.globals import *
from .helpers import datasets

##Configs
BATCH_SIZE = 16

## Example Program Call: train.py "$model" "$lang" "$EPOCHS" "$MAX_SIZE"

def create_tokenize_and_align(tokenizer):
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

    return tokenize_and_align

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

def main():
    if len(sys.argv) != 5:
        print("Invalid Program Call.")
        return
    
    _,model_type,lang_code,epochs,max_size = sys.argv

    # Load Data

    try:
        epochs = int(epochs)
        max_size = int(max_size)
    except:
        print("Epoch/Max_Size cast failed.")
        return
    
    training_set = datasets.get_dataset(lang_code,"train")
    eval_set = datasets.get_dataset(lang_code,"test")

    if max_size != -1:
        training_set = training_set.shuffle(seed=42).select(range(min(len(training_set),max_size)))
    
    # Load Model,Tokenizer, Collator

    pretuned_model = AutoModelForTokenClassification.from_pretrained(
        model2modelname[model_type],
        num_labels=len(UPOS_TAGS),
        label2id=label2id,
        id2label=id2label,
    )

    tokenizer = AutoTokenizer.from_pretrained(model2modelname[model_type])
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    #Tokenize Dataset

    tokenize_and_align = create_tokenize_and_align(tokenizer)

    cols_to_remove = training_set.column_names
    training_set = training_set.map(tokenize_and_align,batched=True,remove_columns=cols_to_remove)
    eval_set = eval_set.map(tokenize_and_align,batched=True,remove_columns=cols_to_remove)

    #Train Model

    training_args = TrainingArguments(
        output_dir=path.join(project_path,f"models/{lang_code}_{model_type}"),
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
        model=pretuned_model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=eval_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return

if __name__ == "__main__":
    main()
