import os
from json import loads as LoadsJson
import logging

from pandas import DataFrame
import torch

from src.utils import writeJsonFile
from src.preprocessing import Paper
from config import directories as Paths

def createRawDataset() -> DataFrame:
    """
    Creates the dataframe for the papers that include the 
    following columns : 
        'paperID', 'abstract', 'relevantSections', 'title'
    """
    _allPapersData : list[dict] = []

    # Iterate through the papers.
    for paperPath in os.listdir(Paths.papers):
        if not paperPath.endswith('.json'):
            continue
        try :
            paper = Paper(f"{Paths.papers}/{paperPath}")
            paperData = paper.getPaperData()
            # Store the data dict for each of the papers.
            _allPapersData.append(paperData)
        except Exception as e:
            logging.error(e)

    dataset = DataFrame(_allPapersData)
    # Making sure that the columns match.
    assert list(dataset.columns) == ['paperID', 'abstract', 'relevantContent', 'title'], "Columns do not match."
    return dataset

def separateOutPaper(paperListFile : str, offset : int ) -> int:
    """
    This function takes the list of papers stored as a list of 
    json objects and then extracts all the individual papars from
    the dataset and stores them in the _papers_ directory.
    
    @param paperListFile: The file containing the list of papers.
    @param offset: The offset to be added to the paper ids.
    @return: The number of papers extracted.
    """
    # Store the number of papers.
    _paperCount = 0
    with open(paperListFile, "r", encoding="utf-8") as file:
        for paperID, line in enumerate(file.readlines()):
            paperData = LoadsJson(line.strip().encode("utf-8"))
            writeJsonFile(f"{Paths.papers}/{paperID + offset}.json", paperData)
            _paperCount += 1

    return _paperCount

def inference(model, input):
    model.eval()
    _, _, qa = model(input)
    # Use the tokenizer to get back the output.
    tokenizer = model.tokenizer
    return tokenizer.decode(qa[0], skip_special_tokens=True)

def formatGoldQnA( qnA : str ):
    "Evaluates the qnA as a list of dicts."
    from ast import literal_eval
    return literal_eval(qnA)

def getMostProbableToken(distribution: torch.Tensor):
    """
    Returns the token index with the highest probability and its corresponding probability.
    """
    token_index = distribution.argmax(dim=-1)
    token_prob = distribution.max(dim=-1).values
    return token_index, token_prob

def getTopKTokens(distribution: torch.Tensor, k: int, normalize: bool = False):
    """
    Returns the top k most probable tokens and their probabilities.
    """
    topKIndices = distribution.argsort(dim=-1)[-k:]
    topKValues = distribution.gather(dim=-1, index=topKIndices)
    
    if normalize:
        topKValues = topKValues / topKValues.sum(dim=-1, keepdim=True)

    return topKIndices, topKValues

def getATokenFromTopK(distribution: torch.Tensor, k: int = 5):
    """
    Returns a weighted random choice from the normalized
    probability distribution of the top k most probable tokens,
    along with the probability of the chosen token.
    """
    topKIndices, topKValues = getTopKTokens(distribution, k, normalize=True)

    if topKValues.sum(dim=-1).to(torch.float32) == 0:
        raise ValueError("All values are 0.")

    # Sample a token index from the top-k indices using the probabilities
    sampled_idx = torch.multinomial(topKValues, num_samples=1).squeeze(-1)
    chosen_token_index = topKIndices.gather(dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)
    chosen_token_prob = topKValues.gather(dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)

    return chosen_token_index, chosen_token_prob
if __name__ == '__main__' :
    # os.makedirs(Paths.papers, exist_ok=True)

    # paperListFileDir = Paths.limitGenData
    # 
    # numPapers = 0 # Setting the number of papers yet done to 0.
    # for paperListFile in os.listdir(paperListFileDir):
    #     numExtractedPapers = separateOutPaper(os.path.join(paperListFileDir, paperListFile), numPapers)
    #     numPapers += numExtractedPapers
    # 
    # print(f"Extracted {numPapers} papers.")
    import pandas as pd
    testData = pd.read_csv(f"{Paths.data}/test.csv")

    print(testData['gold_QnA'][0])
    print(formatGoldQnA(testData['gold_QnA'][0]))

