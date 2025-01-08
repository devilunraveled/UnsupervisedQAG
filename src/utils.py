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
