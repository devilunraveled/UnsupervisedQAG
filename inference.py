"""
This is the code for running inference on the model.
"""
import random
from src.utils import getATokenFromTopK, getMostProbableToken
import time
from eval import compute_rouge_score

if __name__ == "__main__":
    from src.model import Model
    from config import directories as Paths, QnAModel
    import pandas as pd
    import sys

    testData = pd.read_csv(f"{Paths.data}/test.csv")
    modelPath = sys.argv[1]
    # model = Model(modelName = QnAModel.name, inference_mode = True)
    # model.to("cuda")
    # 
    # print(f"Loading model from {modelPath}")
    # model.load_model(modelPath)
    # print(f"Model loaded from {modelPath}")
    # 
    # index = random.randint(0, len(testData))
    # tokenizer = model.tokenizer
    # print(tokenizer.special_tokens_map)
    # inputs = tokenizer(testData['modelPrompt'].iloc[index], return_tensors="pt", truncation=True, max_length=512).to("cuda")
    # 
    # print(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))

    # output = model.inference(
    #     **inputs,
    #     max_length = 256
    # )

    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    model = Model(modelName = QnAModel.name, inference_mode = True)

    print(f"Loading model from {modelPath}")

    model.load_model(modelPath)
    model.to("cuda")

    index = random.randint(0, len(testData))
    maxLength = 256
    tokenizer = model.tokenizer

    try :
        index = int(sys.argv[2])
        maxLength = int(sys.argv[3])
    except Exception :
        print("Using default parameters.")
        pass

    inputs = tokenizer(testData['modelPrompt'].iloc[index], return_tensors="pt", truncation=True, max_length=512).to("cuda")

    print(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))

    output = model.inference(
        **inputs,
        max_length = maxLength,
        sampler = getATokenFromTopK
    )

    tokensTillNow = []
    for intermediate in output :
        tokensTillNow.append(tokenizer.decode(intermediate[0][-1], skip_special_tokens=True))
        print(tokensTillNow[-1], end = " ")
        sys.stdout.flush()
        time.sleep(0.2)

    context = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    modelOutput = ' '.join(tokensTillNow)

    rougeScores = compute_rouge_score(modelOutput, context)
    print(rougeScores)
