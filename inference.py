"""
This is the code for running inference on the model.
"""

from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel

class Inferencer:
    def __init__(self, model):
        self.hf_base_mode_name = model.modelName
        self.custom_model = model

    def load(self, fuse : bool = False):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_base_mode_name)
        if fuse:
            self.load_pre_trained_weights()

    def load_pre_trained_weights(self):
        self.model.encoder = self.custom_model.encoder
        self.model.decoder = self.custom_model.qAGenerationDecoder
        self.model.lm_head = self.custom_model.lmHead_qAGeneration

if __name__ == "__main__":
    from src.model import Model
    from config import directories as Paths, QnAModel
    import pandas as pd
    import sys

    testData = pd.read_csv(f"{Paths.data}/test.csv")
    
    modelPath = sys.argv[1]
    model = Model(modelName = QnAModel.name, inference_mode = True)
    model.load_model(modelPath)
    
    tokenizer = model.tokenizer

    inferencer = Inferencer(model)
    inferencer.load()
    
    inputs = tokenizer.encode(testData['modelPrompt'].iloc[0], return_tensors="pt")
    
    print(tokenizer.decode(inputs[0], skip_special_tokens=True), end = "\n\n")

    output = inferencer.model.generate(
        input_ids = inputs,
        max_length = 1024,
        num_beams = 4
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))
