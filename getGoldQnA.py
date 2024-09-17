from typing import List
from langchain_huggingface import HuggingFacePipeline

from alive_progress import alive_bar

from config import Model
from pandas import DataFrame

def getPipeline():
    return HuggingFacePipeline.from_model_id(model_id = Model.name, task ="text2text-generation",
                                             device = -1)

def paperData( dataset : DataFrame) -> List[str]:
    prompts : List[str] = []
    for paper in dataset.iterrows():
        prompts.append(f"{Model.promptPrefix}{paper[1]['relevantContent']}{Model.promptInfix}{paper[1]['abstract']}{Model.promptSuffix}")
    return prompts

def getResponseForDataset( dataset : DataFrame ):
    pipeline = getPipeline()
    with alive_bar(len(dataset), force_tty = True) as bar:
        for prompt in paperData(dataset):
            print(pipeline(prompt))
            bar()

if __name__ == "__main__":
    from utils import createRawDataset
    dataset = createRawDataset()
    getResponseForDataset(dataset)
