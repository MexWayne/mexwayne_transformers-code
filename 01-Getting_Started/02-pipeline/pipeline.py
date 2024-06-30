# 1 print the support task
"""
from transformers.pipelines import SUPPORTED_TASKS
for k, v in SUPPORTED_TASKS.items():
    print("---------------------------")
    print(k, v)
"""


# 2 build a pipeline
from transformers import pipeline

# case 1 text-classification
model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
text_pipe = pipeline("text-classification", model=model_id)
print("case1:")
print(text_pipe('good!'))

# case 2 object-detect
model_id = "facebook/detr-resnet-50"
objdct_pipe = pipeline("object-detection", model=model_id)
print("case2:")
print(objdct_pipe('/home/mex/Desktop/learn_objdetect/datasets/coco128/images/train2017/000000000025.jpg'))



# case 3 
import torch
import time

model_id = "facebook/detr-resnet-50"
objdct_pipe = pipeline("object-detection", model=model_id)
start = time.time()
for i in range(30):
    objdct_pipe('/home/mex/Desktop/learn_objdetect/datasets/coco128/images/train2017/000000000025.jpg')
end = time.time()
print("case 3:")
print("cpu time:" + str((end - start)))


model_id = "facebook/detr-resnet-50"
objdct_pipe = pipeline("object-detection", model=model_id, device=0) # chose gpu 0
objdct_pipe.model.device
torch.cuda.synchronize()
start = time.time()
for i in range(30):
    objdct_pipe('/home/mex/Desktop/learn_objdetect/datasets/coco128/images/train2017/000000000025.jpg')
torch.cuda.synchronize()
end = time.time()
print("gpu time:" + str((end - start)))





