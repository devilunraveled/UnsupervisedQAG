# from accelerate import Accelerator
from src.trainer import CustomTrainer
from src.model import Model
import pandas as pd
from config import QnAModel, directories as Paths
from torch.utils.data import Dataset
from utils import formatGoldQnA as getQuestionsAsList

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.dataframe['gold_QnA'] = self.dataframe['gold_QnA'].apply(self.formatQnAs)

    def __len__(self) -> int:
        return len(self.dataframe)

    def formatQnAs(self, qnAList) :
        qna_string = ""
        qnAList = getQuestionsAsList(qnAList)
        for qna in qnAList :
            qna_string += f"Q : {qna['question']}\nA : {qna['answer']} {self.tokenizer.sep_token}"
        return qna_string

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        inputs  = self.tokenizer(row['modelPrompt'],    return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        qnAs    = self.tokenizer(row['gold_QnA'],       return_tensors="pt", padding='max_length', truncation=True, max_length=512).input_ids.squeeze()

        inputs['qna_labels'] = qnAs
        return inputs

def train(model, trainData, evalData, trainingArguments, weights = (0.5, 0.5)) :
    # accelerator = Accelerator()
    # optimizer = AdamW(model.parameters(), lr=trainingArguments['learning_rate'])
    # scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                             num_warmup_steps=trainingArguments['warmup_steps'],
    #                                             num_training_steps=len(trainData) * trainingArguments['num_train_epochs'])
    # model, optimizer, trainData, evalData, scheduler = accelerator.prepare(model, optimizer, trainData, evalData, scheduler)

    trainer = CustomTrainer(model, trainData, evalData, trainingArguments, weights = weights)
    trainer.train()

if __name__ == "__main__" :
    squadTrainData = pd.read_csv(f"{Paths.data}/squad.csv")
    trainData = pd.read_csv(f"{Paths.data}/train.csv")
    testData = pd.read_csv(f"{Paths.data}/test.csv")
    
    model = Model(modelName = QnAModel.name, useLORA=True)
    tokenizer = model.tokenizer
    
    squadTrainDataset = CustomDataset(squadTrainData, tokenizer)
    trainDataset = CustomDataset(trainData, tokenizer)
    testDataset = CustomDataset(testData, tokenizer)
    
    training_arguments_squad = {
        'output_dir': QnAModel.outputDirectory,
        'save_steps': 5000,
        'save_total_limit': 5,
        'num_train_epochs': 30,
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 2,
        'eval_strategy': 'no',
        'logging_dir': QnAModel.loggingDirectory,
        'logging_steps': 100,
        'learning_rate': 2e-5,
        'warmup_steps': 50,
        'deepspeed' : 'ds_config.json',
    }
    
    # train(model, squadTrainDataset, testDataset, trainingArguments = training_arguments_squad, weights = (0.3,0.7))
    # print("Model fine-tuned on Squad Dataset, saving model")
    # model.save_model('/scratch/jai.bhatnagar/hardik_uqag/bart_large_ft_squad_30/')

    training_arguments = {
        'output_dir': QnAModel.outputDirectory,
        'num_train_epochs': 10,
        'save_steps': 5000,
        'save_total_limit': 5,
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 2,
        'eval_strategy': 'no',
        'logging_dir': QnAModel.loggingDirectory,
        'logging_steps': 50,
        'learning_rate': 1e-4,
        'deepspeed' : 'ds_config.json',
    }
    
    train(model, trainDataset, testDataset, trainingArguments = training_arguments, weights = (0.3,0.7))
    print("Training finished, saving model")
    model.save_model('/scratch/jai.bhatnagar/hardik_uqag/bart_large_ft_squad_30_ft_arxiv_10/')
