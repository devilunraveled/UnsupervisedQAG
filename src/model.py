"""
The model architecture for the Encoder-2*Decoder model.
"""
from typing import override
from torch import nn as NeuralNetwork

class Model(NeuralNetwork.Module):
    def __init__(self, encoder, reconstructionDecoder, qAGenerationDecoder) :
        super(Model, self).__init__()
        self.encoder = encoder
        self.reconstructionDecoder = reconstructionDecoder
        self.qAGenerationDecoder = qAGenerationDecoder
    
    @override
    def forward(self, input) :
        encoderOutput = self.encoder(input)
        reconstructionOutput = self.reconstructionDecoder(encoderOutput)
        qAGenerationOutput = self.qAGenerationDecoder(encoderOutput)
        return encoderOutput, reconstructionOutput, qAGenerationOutput
