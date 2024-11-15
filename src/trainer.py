from transformers import Trainer, TrainingArguments
from torch import nn as NeuralNetwork, argmax, tensor
from .utils import shiftLabels
from torch.amp import autocast

class CustomTrainer(Trainer) :
    def __init__(self, model, trainData, evalData, trainingArguments, weights = (0.5, 0.5) ) :
        super().__init__(model=model, train_dataset = trainData, eval_dataset = evalData, args = TrainingArguments(**trainingArguments))
        self.tokenizer = model.tokenizer
        self.weights = weights
        self.lossFunction = NeuralNetwork.CrossEntropyLoss(ignore_index = self.tokenizer.pad_token_id)

    def compute_loss(self, model, inputs, return_outputs=False):
        qna_labels = inputs.pop("qna_labels")
        input_ids = inputs.pop("input_ids").squeeze(1)
        reconstruction_labels = input_ids.clone()
        attn_mask = inputs.pop("attention_mask").squeeze(1)
        
        padToken = self.tokenizer.pad_token_id
        startToken = self.tokenizer.pad_token_id
        
        outputs = model(input_ids = input_ids, 
                        attention_mask = attn_mask, 
                        qna_labels = shiftLabels(qna_labels, padToken, startToken), 
                        reconstruction_labels = reconstruction_labels)
        
        reconstruction_output = outputs[1]
        qna_output = outputs[2]
        vocabSize = outputs[2].shape[-1]
        
        # print(self.tokenizer.decode(input_ids[0]))
        # print("Reconstruction Labels :")
        # print(self.tokenizer.decode(reconstruction_labels[0]))
        # print("QnA Labels :")
        # print(self.tokenizer.decode(qna_labels[0]))
        # print("Model Output:")
        # print(self.tokenizer.decode(reconstruction_output[0].argmax(dim = -1)))

        def customLossFunction(output1, output2, input1, input2, weight1, weight2) :
            output1 = output1.view(-1, vocabSize)
            output2 = output2.view(-1, vocabSize)
            input1 = input1.view(-1)
            input2 = input2.view(-1)
            loss1 = self.lossFunction(output1, input1) 
            loss2 = self.lossFunction(output2, input2)
            
            with autocast(device_type="cuda"):
                loss = loss1 * weight1 + loss2 * weight2
                return loss

        loss = customLossFunction(reconstruction_output, qna_output, qna_labels, reconstruction_labels, self.weights[0], self.weights[1])

        if return_outputs:
            return (loss, outputs)
        else:
            return loss


