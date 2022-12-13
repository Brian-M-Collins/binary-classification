# %%
from datasets import load_dataset
import awswrangler as wr

# %%
training_data = wr.s3.read_csv("s3://wileyintsoljupyterhub/bluebird/bcollins/balanced_training_set.csv").drop('Unnamed: 0', axis=1)
testing_data = wr.s3.read_csv("s3://wileyintsoljupyterhub/bluebird/bcollins/balanced_test_set.csv").drop('Unnamed: 0', axis=1)
validation_data = wr.s3.read_csv("s3://wileyintsoljupyterhub/bluebird/bcollins/balanced_validation_set.csv").drop('Unnamed: 0', axis=1)

# %%
data_files = {
    "train":"/workspaces/bluebird/notebooks/brian/generic_review_detection/data/balanced_training_set.csv", 
    "test":"/workspaces/bluebird/notebooks/brian/generic_review_detection/data/balanced_test_set.csv", 
    "validation":"/workspaces/bluebird/notebooks/brian/generic_review_detection/data/balanced_validation_set.csv"
}
dataset = load_dataset('csv', data_files=data_files).remove_columns('Unnamed: 0')

# %%
from transformers import AutoTokenizer
 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# %%
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %%
from transformers import AutoModelForSequenceClassification
checkpoint = "distilbert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# %%
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=1)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
