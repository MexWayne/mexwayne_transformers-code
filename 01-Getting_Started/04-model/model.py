from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
os.environ['CURL_CA_BUNDLE'] = ''


if __name__ == "__main__":

    # down load model, the configuration of a model 
    model = AutoModel.from_pretrained("hfl/rbt3")
    #print(model.config)

    # save the model
    #git lfs clone "https://huggingface.co/hfl/rbt3" --include="*.bin"
    
    # config a model
    config = AutoConfig.from_pretrained("./rbt3/")
    #print(config) # same with print model
    

    # create a input by token
    text = "我爱吃羊肉泡馍和肉夹馍!"
    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    inputs = tokenizer(text, return_tensors="pt")
    #print(inputs)

    # model inference
    model = AutoModel.from_pretrained("hfl/rbt3", output_attentions=True)
    output1 = model(**inputs)
    print("model inference:") 
    print(output1)
    # get the tensor 
    print(output1.last_hidden_state.size())
    
    # model head inference
    from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
    clz_model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3", num_labels=10)
    output2 = clz_model(**inputs)
    print("model head inference:") 
    print(output2)

