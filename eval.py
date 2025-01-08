from rouge_score import rouge_scorer

from src.utils import getATokenFromTopK, getMostProbableToken
import time
from eval import compute_rouge_score
import random

def compute_rouge_score(reference: str, hypothesis: str):
    """
    Compute ROUGE scores between a reference text and a hypothesis text.

    Args:
        reference (str): The reference or ground truth text.
        hypothesis (str): The generated or predicted text.

    Returns:
        dict: A dictionary with ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'ROUGE-1': scores['rouge1'],
        'ROUGE-2': scores['rouge2'],
        'ROUGE-L': scores['rougeL']
    }


if __name__ == "__main__":
    from src.model import Model
    from config import directories as Paths, QnAModel
    import pandas as pd
    import sys

    testData = pd.read_csv(f"{Paths.data}/test.csv")
    modelPath = sys.argv[1]
    
    model = Model(modelName = QnAModel.name, inference_mode = True)

    print(f"Loading model from {modelPath}")

    model.load_model(modelPath)
    model.to("cuda")

    index = random.randint(0, len(testData))
    maxLength = 256
    tokenizer = model.tokenizer
    
    def getResponse(inputs) : 
        return model.inference(
            **inputs,
            max_length = maxLength,
            sampler = getATokenFromTopK
        )
    
    for index in range(1,50):
        inputs = tokenizer(testData['modelPrompt'].iloc[index], return_tensors="pt", truncation=True, max_length=512).to("cuda")

        modelOutput = tokenizer.decode( getResponse(inputs)[0], skip_special_tokens=True)
        context = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        rougeScores = compute_rouge_score(modelOutput, context)
        print(rougeScores)
    
