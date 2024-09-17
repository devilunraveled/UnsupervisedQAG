## Implement a vLLM pipeline to get batched inferencing.
## Link :https://docs.vllm.ai/en/latest/getting_started/quickstart.html

from pandas import DataFrame
from vllm import LLM, SamplingParams

from alive_progress import alive_bar
from config import Model

def instantiate_model(temperature=0.4, top_p=0.95):
    samplingParams = SamplingParams(temperature=temperature, top_p=top_p)
    model = LLM(model=Model.name, sampling_params=samplingParams)
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

if __name__ == '__main__':
    from utils import createRawDataset
    dataset = createRawDataset()
    listOfQnA = getQnAForData(dataset)
