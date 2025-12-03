import sys,numpy as np,os,json
from transformers import Trainer,AutoTokenizer,DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, f1_score

from .helpers.globals import *
from .helpers import models,datasets

label_list = [id2label[i] for i in range(len(id2label))]
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

def main():
    _, model_type, trained_lang, eval_lang = sys.argv

    model = models.get_trained_model(trained_lang,model_type)
    eval_set = datasets.get_dataset(eval_lang,"test")

    tokenizer = AutoTokenizer.from_pretrained(model2modelname[model_type])
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        eval_dataset=eval_set.map(create_tokenize_and_align(tokenizer),batched=True,remove_columns=eval_set.column_names),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
      
    res = trainer.evaluate()

    results_name = f"{model_type}_{trained_lang}_{eval_lang}.json"

    res_dir = os.path.join(project_path,"results")

    os.makedirs(res_dir,exist_ok=True)
    with open(os.path.join(res_dir,results_name),"w") as f:
        f.write(json.dumps(res))







if __name__ == "__main__":
     main()
