import logging
from json import load as LoadJson, dump as DumpJson

import numpy as np
from numpy import ndarray as NDArray
import numpy.random as random
import torch

def readJsonFile(filePath : str):
    try :
        with open(filePath, "r", encoding="utf-8") as file:
            fileData = LoadJson(file)
        logging.info(f"File read : {filePath}")
    except FileNotFoundError:
        fileData = None
        logging.error(f"File not found : {filePath}")
    return fileData

def writeJsonFile(filePath : str, data : dict):
    try :
        with open(filePath, "w", encoding="utf-8") as file:
            DumpJson(data, file)
        logging.info(f"File written : {filePath}")
    except FileNotFoundError:
        logging.error(f"File not found : {filePath}")

def shiftLabels(labels, pad_token_id, start_token_id):
    shifted_labels = labels.new(labels.size()).fill_(pad_token_id)
    shifted_labels[..., 1:] = labels[..., :-1].clone()
    shifted_labels[..., 0] = start_token_id
    return shifted_labels


############ SAMPLING TECHNIQUES ############

def getMostProbableToken( distribution : NDArray ) :
    """
    Returns the token index with the highest probabilty. 
    """
    return distribution.argmax(axis = -1)

def getATokenFromTopK( distribution : NDArray, k : int) :
    """
    Returns a weighted random choice from the normalized
    probability distribution from the top k most 
    probable tokens.
    """
    topKIndices, topKValues = getTopKTokens(distribution, k)
    if topKValues.sum(axis = -1, dtype = 'float32') == 0 :
        raise Exception("All values are 0.")

    topKValues = topKValues / topKValues.sum(axis = -1, dtype = 'float32')
    return random.choice(a = topKIndices, p = topKValues, replace = False)

def getTopKTokens( distribution : NDArray , k : int, normalize : bool = False ) :
    """
    Returns the top k most probable tokens.
    """
    topKIndices = distribution.argsort(axis=-1)[..., -k:]
    topKValues = np.take_along_axis(distribution, topKIndices, axis=-1)
    if normalize :
        topKValues = topKValues / topKValues.sum(axis = -1, dtype = 'float32')
    
    return topKIndices, topKValues

def infer(model, input):
    with torch.no_grad():
        padToken = model.tokenizer.pad_token_id
        startToken = model.tokenizer.pad_token_id

        qna_labels = shiftLabels(input.pop("qna_labels"), padToken, startToken)
        reconstruction_labels = shiftLabels(input.pop("reconstruction_labels"), padToken, startToken)

        input_ids = input.pop("input_ids").squeeze(1)
        attn_mask = input.pop("attention_mask").squeeze(1)

        _, _, qa = model(input_ids = input_ids, 
                        attention_mask = attn_mask, 
                        qna_labels = qna_labels.unsqueeze(0), 
                        reconstruction_labels = reconstruction_labels.unsqueeze(0))

        print(qa.shape)
        return model.tokenizer.decode(qa[0], skip_special_tokens=True)
