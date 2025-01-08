# from accelerate import Accelerator
from src.trainer import CustomTrainer
from src.model import Model
import pandas as pd
from config import QnAModel, RESEARCHTrainingConfig, SQUADTrainingConfig, directories as Paths, HyperParams, QuantizationConfig
from torch.utils.data import Dataset
from utils import formatGoldQnA as getQuestionsAsList
# from accelerate import Accelerator

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
    # accelerator = Accelerator()
    squadTrainData = pd.read_csv(f"{Paths.data}/squad.csv")
    trainData = pd.read_csv(f"{Paths.data}/train.csv")
    testData = pd.read_csv(f"{Paths.data}/test.csv")
    
    squadTrainData = squadTrainData.reset_index(drop=True)
    trainData = trainData.reset_index(drop=True)
    testData = testData.reset_index(drop=True)
    
    squadTrainData = squadTrainData.drop(columns=["Unnamed: 0"], errors='ignore')
    trainData = trainData.drop(columns=["Unnamed: 0"], errors='ignore')
    testData = testData.drop(columns=["Unnamed: 0"], errors='ignore')
    
    totalDataset = pd.concat([trainData, squadTrainData], axis = 0, ignore_index = True)
    print(f"Total dataset size : {len(totalDataset)}")
    
    model = Model(modelName = QnAModel.name, useLORA=True, bitsAndBytesConfig = QuantizationConfig)
    tokenizer = model.tokenizer 

    squadTrainDataset = CustomDataset(squadTrainData, tokenizer)
    trainDataset = CustomDataset(trainData, tokenizer)
    testDataset = CustomDataset(testData, tokenizer)
    
    squad_training_arguments = {**SQUADTrainingConfig}
    research_training_arguments = {**RESEARCHTrainingConfig}

    modelSize = 'large' if 'large' in QnAModel.name else 'base'

    SQUADTrainer = CustomTrainer(model, squadTrainDataset, testDataset, squad_training_arguments, weights = HyperParams['squad_fine_tine_weights'])
    try :
        SQUADTrainer.train()
        print("Training finished on SQUAD, saving model")
    finally :   
        model.save_model(f'/scratch/jai.bhatnagar/hardik_uqag/flan-t5-{modelSize}-squad/')

    RESEARCHTrainer = CustomTrainer(model, trainDataset, testDataset, research_training_arguments, weights = HyperParams['research_fine_tine_weights'])
    try :
        RESEARCHTrainer.train()
        print("Training finished on RESEARCH Dataset, saving model")
    finally :   
        model.save_model(f'/scratch/jai.bhatnagar/hardik_uqag/flan-t5-{modelSize}-squad-research-{HyperParams["research_fine_tine_weights"][0]*10}')
