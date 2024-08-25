from transformers import AutoTokenizer

if __name__ == "__main__":
    sentense = "我爱吃羊肉泡馍和肉夹馍!"
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
    print(tokenizer)

    # save the tokenizer to local path
    tokenizer.save_pretrained("./roberta_tokenizer")

    # load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer/")

    # luanch the tokenizer
    tokens = tokenizer.tokenize(sentense)
    print(tokens)

    # show the vecab
    #print(tokenizer.vocab)
    #print(tokenizer.vocab_size)

    # convert the tokens into ids 
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    # convert ids into tokens
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(tokens)

    # convert the token into sentense
    str_sentense = tokenizer.convert_tokens_to_string(tokens)
    print(str_sentense)



    # encoder
    ids = tokenizer.encode(sentense, add_special_tokens=True)
    print(ids)
    # decoder 
    str_sentense = tokenizer.decode(ids, skip_special_tokens=False)
    print(str_sentense)

    # clip
    ids = tokenizer.encode(sentense, max_length=5, truncation=True)
    print("clip:" + str(ids))
    # fill
    ids = tokenizer.encode(sentense, padding="max_length", max_length=20)
    print("fill:" + str(ids))

    #mask
    attention_mask = [1 if idx != 0 else 0 for idx in ids]
    token_type_ids = [0] * len(ids)
    print(ids)
    print(attention_mask)
    print(token_type_ids)


    # advanced tokenizer
    inputs = tokenizer.encode_plus(sentense, padding="max_length", max_length=20)
    print(inputs)
    inputs = tokenizer(sentense, padding="max_length", max_length=20)
    print(inputs)


    # batchh
    import time
    print("batch test:")
    text = ["我爱吃泡馍",
            "吃泡馍前要先洗手然后用手掰馍",
            "泡馍可以有羊肉泡馍和牛肉泡馍",
            "吃泡馍时还要配有糖蒜",
            "糖蒜可以解腻",
            "吃完泡馍可以要一碗高汤"]
    start = time.time()
    tokens = tokenizer(text, padding="max_length", max_length=20)
    end = time.time()
    print("deal with batch:" + str(end - start))
    # loop
    start = time.time()
    for i in range(len(text)):
            tokens = tokenizer(text[i], padding="max_length", max_length=20)
    end = time.time()
    print("deal with loop :" + str(end - start))


    # slow
    import time
    start = time.time()
    for i in range(100):
        fast_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
    end = time.time()
    print("fast cost time:" + str(end - start))
    # slow
    start = time.time()
    for i in range(100):
        slow_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", use_fast=False)
    end = time.time()
    print("slow cost time:" + str(end - start))
    
