from transformers import AutoTokenizer, AutoModelForSequenceClassification  # tokenizer and model will help text classiffication
import pandas as pd # used for dealing with csv data
from torch.utils.data import Dataset
#from torch.utils.data import random_split # my torch version is 1.12.0 this functions is added at lest 1.13, so I add it in utils.py
import torch
import utils as ut

from torch.utils.data import DataLoader





class TxtDataset(Dataset):
    # return None type value import code readability
    def __init__(self) -> None:
        super().__init__()
        # read the data
        self.data = pd.read_csv("/home/mex/Desktop/learn_transformer/mexwayne_transformers_NLP/01-Getting_Started/04-model/ChnSentiCorp_htl_all.csv")
        # remove null data
        self.data = self.data.dropna()

    def __getitem__(self, index): # return one label with data in one time
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self): #return the size of current dataset 
        return len(self.data)
    

# when we train the model, we will make traindata with batchsize and pack them with into one tensor
# so we need implement a function like this
# the text and label should be store.    
def collate_fun(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    # the datat which is too long should be truncated with max_length, and the short data should be 
    # padd all the data into same length
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    # if the label in the data, the loss will caculate by model like brt did so
    inputs["labels"] = torch.tensor(labels)
    return inputs

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch) # input the valid data into model and get the reuslts
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum() # the value after == is type of bool, we need to change it into int
    return acc_num / len(validset) # finaly we count the negtive and positive value 


def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train() # need to open the train mode for model
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad() # the optimizer should be set zero first before used
            output = model(**batch) # we want to put all the key into it, so we use double start
            print(output)
            output.loss.backward()
            optimizer.step() # update the model
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1 # gobale step need increase
        acc = evaluate() # check the accuracy
        print(f"ep: {ep}, acc: {acc}")




if __name__ == "__main__":

    txtdataset = TxtDataset()
    for i in range(5):
        # print the sentense with label, 1 shows positive comment
        print(txtdataset[i])
    
    print("len txtdataset:" + str(len(txtdataset)))

    # length means the propotion, 
    # for this case, train dataset is 0.9 and valid dataset is 0.1, 
    # validset + tran must equals 1.0
    trainset, validset = ut.random_split(txtdataset, [.9, .1], generator=torch.Generator().manual_seed(42))
    print("len trainset:" + str(len(trainset)))
    print("len validset:" + str(len(validset)))

    for i in range(10):
        print(trainset[i])

    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True,  collate_fn=collate_fun)
    validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_fun)

    print(type(trainloader))


    from torch.optim import Adam # define the trainer, use adam gradiant descent
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    if torch.cuda.is_available(): # model should be set on the gpu
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=2e-5)

    train()

    
    sen = "我觉得这家羊肉泡馍馆子牛逼，做的羊肉牛逼！"
    id2_label = {0: "差评！", 1: "好评！"}
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(sen, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        print(f"输入：{sen}\n模型预测结果:{id2_label.get(pred.item())}")