from typing import override
from transformers import Trainer, TrainingArguments
from torch import nn as NeuralNetwork

class CustomTrainer(Trainer) :
    def __init__(self, model, trainData, evalData, **trainingArguments) :
        super(Trainer, self).__init__()

        self.trainingArguments = TrainingArguments(**trainingArguments)
        self.model = model
        self.trainData = trainData
        self.evalData = evalData
    
    @override
    def compute_loss(self, model, inputs, return_outputs=False):
        labels1 = inputs.pop("labels1")
        labels2 = inputs.pop("labels2")

        outputs = model(**inputs)
        
        if not return_outputs :
            return customLossFunction(outputs[0], outputs[1], labels1, labels2, 0.5, 0.5)

        return outputs, customLossFunction(outputs[0], outputs[1], labels1, labels2, 0.5, 0.5)

def customLossFunction(output1, output2, input1, input2, weight1, weight2) :
    lossFunction = NeuralNetwork.CrossEntropyLoss()
    loss1 = lossFunction(output1, input1)
    loss2 = lossFunction(output2, input2)
    return loss1 * weight1 + loss2 * weight2
