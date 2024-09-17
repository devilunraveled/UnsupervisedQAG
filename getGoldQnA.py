## Implement a vLLM pipeline to get batched inferencing.
## Link :https://docs.vllm.ai/en/latest/getting_started/quickstart.html

from pandas import DataFrame
from alive_progress import alive_bar
from config import Model, directories as Paths
import os

def instantiate_model(temperature=0.4, top_p=0.95):
    from vllm import LLM, SamplingParams
    samplingParams = SamplingParams(temperature=temperature, top_p=top_p)
    model = LLM(model=Model.name, dtype = 'float16', quantization='gptq')
    return model, samplingParams

def getGoldQnA(batch, model, samplingParams):
    return model.generate(batch, samplingParams)

def getQnAForData( dataset : DataFrame, batchSize : int = 1) -> list:
    """
    Returns the QnA generated using the model in the dataset.
    The batchSize argument can be passed to see the model inference latency.
    """
    model, samplingParams = instantiate_model()
    
    def batched(data, batchSize = 1):
        for i in range(0, len(data), batchSize):
            yield data[i:i+batchSize]

    listOfQnA = []
    totalBatches = (len(dataset['relevantContent']) // batchSize) + (1 if len(dataset['relevantContent']) % batchSize > 0 else 0)
    with alive_bar(totalBatches, length = 20, title = "Generating QnA") as bar:
        for batch in batched(dataset['relevantContent']):
            listOfQnA.append(getGoldQnA(batch, model, samplingParams))
            print(listOfQnA[-1])
            bar.update()
    return listOfQnA

def useLangchainOllama(papers : DataFrame) -> None:
    """
    Saves the QnA in the _papers_ directory, by the model.
    All progress is saved, so can cacncel the operation
    """
    from langchain_community.llms import Ollama
    from src.postProcessing import extractQAPairs
    import pickle as Pickle

    model = Ollama(model=Model.name)
    
    with alive_bar(len(papers), length = 20, title = "Generating QnA") as bar:
        for _, paperData in papers.iterrows():
            goldQnA = None
            try :
                if os.path.exists(f"{Paths.limitGenData}/{paperData['paperID']}.pkl"):            
                    bar()
                    continue
                prompt = f"{Model.promptPrefix} {paperData['relevantContent']} {Model.promptInfix} {paperData['title']} {Model.promptSuffix}"
                modelOutput = model(prompt)
                goldQnA = extractQAPairs(modelOutput)
            except Exception as e:
                print(e)
            finally :
                if goldQnA is not None:
                    with open(f'{Paths.papers}/{paperData["paperID"]}.pkl', 'wb') as f:
                        Pickle.dump(goldQnA, f)
                
if __name__ == '__main__':
    from utils import createRawDataset
    dataset = createRawDataset()
    useLangchainOllama(dataset)
