import evaluate

if __name__ == "__main__":

    ######################################################## basic learn 
    # see the evalution function that evaluate support
    #print(evaluate.list_evaluation_modules())
    #print(evaluate.list_evaluation_modules(include_community=False, with_details=True))

    # load the accuracy class 
    # accuracy = evaluate.load("accuracy")
    # introduce the accuracy functions
    # print(accuracy.description)

    # the inputs help guide 
    # print(accuracy.inputs_description)


    ######################################################## inputs and call 
    ## global accuracy
    #accuracy = evaluate.load("accuracy")
    #results = accuracy.compute(references=[0,1,2,0,1,2], predictions=[0,1,1,2,1,0])
    #print("global way:")
    #print(results)

    ## iterate accurarcy
    #accuracy = evaluate.load("accuracy")
    #for refs, preds in zip([0,1,2,3,4],[0,1,2,3,3]):
    #    accuracy.add(references=refs, predictions=preds)
    #print("iterate way:")
    #print(accuracy.compute())

    ## batch iterate accurarcy
    #accuracy = evaluate.load("accuracy")
    #for refs, preds in zip([[0,1,2,3,4], [0,1,2,3,3], [8,8,8,8,8]],  # refs batches
    #                       [[0,1,2,3,4],[0,1,2,3,5], [8,8,8,8,8]]):  # preds batches
    #    accuracy.add_batch(references=refs, predictions=preds)
    #print("batch iterate way:")
    #print(accuracy.compute())


    # multiple labels
    #clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])
    #print(clf_metrics.compute(predictions=[0, 1, 1, 1, 1], references=[0, 1, 0, 1, 1]))

     
    ##################################################### visual
    from evaluate.visualization import radar_plot
    data = [
        {"accuracy": 0.99, "precision": 0.80, "f1": 0.95, "latency_in_seconds": 33.6 , "recall":0.5},
        {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2 , "recall":0.5},
        {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6 , "recall":0.6}, 
        {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6, "recall":0.7},
        {"accuracy": 0.78, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 100.0, "recall":0.9}
    ]
    model_names = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]
    plot = radar_plot(data=data, model_names=model_names)
    print(type(plot))
    plot.show()
    plot.savefig('radar.png')