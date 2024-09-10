# if the py name is datasets, the import action will first use the current file 
# not the datasets installed by pip
# for example you may meet the error: will be "NameError: name 'load_dataset' is not defined"

from datasets import *
from transformers import AutoTokenizer

# add a prefix for every title
def add_prefix(example):
    example["title"] = 'Prefix: ' + example["title"]
    return example

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
def preprocess_function(example, tokenizer = tokenizer):
    model_inputs = tokenizer(example["content"], max_length = 512, truncation = True)
    labels = tokenizer(example["title"], max_length=32, truncation=True)
    # en:label is title coding results # cn:label 就是 title 的 编码结果
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":


    ####################################################### basic operation
    ## add a dataset
    #data_set = load_dataset("madao33/new-title-chinese")
    #print(data_set)
    #print("------------------------------")
    #print("train[0]:")
    #print(data_set["train"][0])
    #print("------------------------------")
    #print("train[:2]:")
    #print(data_set["train"][:2])
    #print("------------------------------")
    #print("train[\"tile\"][:5]:")
    #print(data_set["train"]["title"][:5])
    #print("------------------------------")




    ####################################################### load specific part or implement
    ## add specific task
    #boolq_dataset = load_dataset("super_glue", "boolq",trust_remote_code=True)
    #print(boolq_dataset)


    #dataset = load_dataset("madao33/new-title-chinese", split="train")
    #print("train:") 
    #print(dataset)

    #dataset = load_dataset("madao33/new-title-chinese", split="train[10:100]")
    #print("train 10:100:") 
    #print(dataset)
    
    #dataset = load_dataset("madao33/new-title-chinese", split="train[10%:50%]")
    #print("train 10%:100%:") 
    #print(dataset)

    #dataset = load_dataset("madao33/new-title-chinese", split=["train[:40%]", "train[40%:]"])
    #print("train 40% and 60%:") 
    #print(dataset)


    ######################################################### split
    #datasets = load_dataset("madao33/new-title-chinese")
    #print("origin train datasets:")
    #print(datasets["train"])
    #print("-----------------")
    #print("make train set as test 0.1:")
    #dataset = datasets["train"]
    #print(dataset.train_test_split(test_size=0.1))
    #print("-----------------")
    #print("stratify:")
    #boolq_dataset = load_dataset("super_glue", "boolq",trust_remote_code=True)
    #dataset = boolq_dataset["train"]
    #print(dataset.train_test_split(test_size=0.1, stratify_by_column="label"))     # 分类数据集可以按照比例划分
    #print("-----------------")

    ########################################### select and filter
    #datasets = load_dataset("madao33/new-title-chinese")
    ## 选取
    #filter_res = datasets["train"].select([0, 1])
    #print("select:")
    #print(filter_res["title"][:5])
    ## 过滤
    #filter_dataset = datasets["train"].filter(lambda example: "中国" in example["title"])
    #print("filter:")
    #print(filter_dataset["title"][:5])
    
    
    ######################################### mapping
    #datasets = load_dataset("madao33/new-title-chinese")
    #prefix_dataset = datasets.map(add_prefix)
    #print(prefix_dataset["train"][:10]["title"])

    #processed_datasets = datasets.map(preprocess_function)
    #print("train:")
    #print(processed_datasets["train"][:5])
    #print("validation:")
    #print(processed_datasets["validation"][:5])

    ######################################## load and save
    datasets = load_dataset("madao33/new-title-chinese")
    processed_datasets = datasets.map(preprocess_function)
    print("from web:") 
    print(processed_datasets["validation"][:2])
    processed_datasets = datasets.map(preprocess_function)
    processed_datasets.save_to_disk("./processed_data")
    processed_datasets = load_from_disk("./processed_data")
    print("from local:") 
    print(processed_datasets["validation"][:2])

