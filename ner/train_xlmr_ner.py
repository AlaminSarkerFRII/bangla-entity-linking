import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

MODEL_NAME = "xlm-roberta-base"
DATA_PATH = "data/processed/silver_ner_dataset.json"

label_list = ["O", "B-ENT", "I-ENT"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# Load data
data = json.load(open(DATA_PATH, encoding="utf-8"))
dataset = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    word_ids = tokenized.word_ids()
    labels = []
    prev_word = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word:
            labels.append(label2id[example["labels"][word_id]])
        else:
            labels.append(-100)
        prev_word = word_id

    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(tokenize_and_align_labels)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)

    true_preds = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]

    return metric.compute(predictions=true_preds, references=true_labels)

args = TrainingArguments(
    output_dir="outputs/ner",
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(100)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
