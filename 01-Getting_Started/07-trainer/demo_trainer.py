# import the related package
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

import torch
import evaluate
from transformers import DataCollatorWithPadding


def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples


def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc


if __name__ == "__main__":
    # download the dataset
    dataset = load_dataset("csv", data_files="/home/mex/Desktop/learn_transformer/mexwayne_transformers_NLP/01-Getting_Started/07-trainer/ChnSentiCorp_htl_all.csv", split="train")
    dataset = dataset.filter(lambda x: x["review"] is not None)
    print("load dataset:")
    print(dataset)

    # split the dataset into 0.1 and 0.9, the 0.1 for test_dataset
    datasets = dataset.train_test_split(test_size=0.1)
    print("split dataset:")
    print(dataset)

    # build tokenize the data
    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
    print(tokenized_datasets)

    # load the model
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    print("model config:")
    print(model.config)

    # build the evaluate module
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")


    # build train args 
    train_args = TrainingArguments(output_dir="./checkpoints",  # train model output path 
                               per_device_train_batch_size=64,  # train batch_size
                               per_device_eval_batch_size=128,  # test batch_size
                               logging_steps=10,                # log 
                               eval_strategy="epoch",     # evaluation strategy
                               save_strategy="epoch",           # save the model every epoch 
                               save_total_limit=3,              # only keep 3 model save
                               learning_rate=2e-5,              #  
                               weight_decay=0.01,               #  
                               metric_for_best_model="f1",      #  
                               load_best_model_at_end=True)     # save the best model after train 
    print("train_args")
    print(train_args)


    # build trainer
    trainer = Trainer(model=model, 
                      args=train_args, 
                      train_dataset=tokenized_datasets["train"], 
                      eval_dataset=tokenized_datasets["test"], 
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=eval_metric)
    
    # start trainnig
    #trainer.train()

    # evaluation
    #trainer.evaluate(tokenized_datasets["test"])
    #####################################################


    #####################################################
    # try one new, to predict a results
    trainer.predict(tokenized_datasets["test"])
    from transformers import pipeline
    
    model.config.id2label = id2_label
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
    sen = "我觉得还是蛮不错的哈！"
    print(pipe(sen))