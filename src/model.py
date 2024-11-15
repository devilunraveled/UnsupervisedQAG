"""
The model architecture for the Encoder-2*Decoder model with QLoRA.
"""
from torch import nn as NeuralNetwork, softmax as Softmax, save as Save, load as Load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, save_peft_model, PeftModel
from copy import deepcopy
from safetensors.torch import load_file
from transformers.modeling_utils import load_state_dict


class Model(NeuralNetwork.Module):
    def __init__(self, modelName, inference_mode : bool = False, useLORA : bool = False, bitsAndBytesConfig = None):
        super(Model, self).__init__()

        self.modelName = modelName
        # Load tokenizer
        self.tokenizer = self.get_tokenizer()
        self.tokenizer.sep_token = '[qna-end]\n'
        
        if bitsAndBytesConfig is not None:
            self.quantizationConfig = BitsAndBytesConfig(**bitsAndBytesConfig)
        else :
            self.quantizationConfig = BitsAndBytesConfig()
        
        # Load encoder and decoders
        self.encoder, self.reconstructionDecoder, self.qAGenerationDecoder, self.lmHead_reconstruction, self.lmHead_qAGeneration = self.getEncoderDecoders(modelName)
        self.model_dimension = self.encoder.config.d_model
        
        # # Print names of encoder modules
        # print("Encoder modules:")
        # for name, _ in self.encoder.named_modules():
        #     print(name)
        # 
        # # Print names of decoder modules
        # print("Reconstuction Decoder modules:")
        # for name, _ in self.reconstructionDecoder.named_modules():
        #     print(name)
        #
        # # Print names of decoder modules
        # print("QnA Decoder modules:")
        # for name, _ in self.qAGenerationDecoder.named_modules():
        #     print(name)
        
        if inference_mode:
            return

        totalTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {totalTrainableParams}")

        # Prepare for k-bit training (e.g., 4-bit)
        # self.encoder = prepare_model_for_kbit_training(self.encoder)
        # self.reconstructionDecoder = prepare_model_for_kbit_training(self.reconstructionDecoder)
        # self.qAGenerationDecoder = prepare_model_for_kbit_training(self.qAGenerationDecoder)
        

        # Configure LoRA for the three modules
        if useLORA:
            self.encoder = self.configure_lora(self.encoder, encoder = True)
            self.reconstructionDecoder = self.configure_lora(self.reconstructionDecoder)
            self.qAGenerationDecoder = self.configure_lora(self.qAGenerationDecoder)
            
            loRATrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total number of trainable parameters with LoRA: {loRATrainableParams}")

            print(f"Percentage Reduction in Training Params : {(1 - loRATrainableParams/totalTrainableParams) * 100} %")

    def getEncoderDecoders(self, modelName):
        model = AutoModelForSeq2SeqLM.from_pretrained(modelName).model
        encoder = model.encoder
        
        reconstructionDecoder = deepcopy(model.decoder)
        qAGenerationDecoder = deepcopy(model.decoder)
        
        # Randomize the lm_head
        model.lm_head = NeuralNetwork.Linear(model.config.d_model, model.config.vocab_size)

        lmHead_reconstruction = deepcopy(model.lm_head)
        lmHead_qAGeneration = deepcopy(model.lm_head)

        return encoder, reconstructionDecoder, qAGenerationDecoder, lmHead_reconstruction, lmHead_qAGeneration
    
    def configure_lora(self, module, encoder : bool = False):
        taskType = 'Seq2SeqLM'
        
        targetModules = [ moduleName for moduleName, _ in module.named_modules() if moduleName != '' and moduleName.endswith('.k') or moduleName.endswith('.q') ]

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            # target_modules=targetModules,
            task_type=taskType,
        )
        return get_peft_model(module, peft_config)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.modelName, clean_up_tokenization_spaces = True)

    def forward(self, input_ids, attention_mask, qna_labels, reconstruction_labels):
        encoderOutput = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        lastHiddenStates = encoderOutput.last_hidden_state
        
        # Forward through both decoders
        qAGenerationDecoderOutput = self.qAGenerationDecoder(
            input_ids = qna_labels,
            encoder_hidden_states = lastHiddenStates,
            encoder_attention_mask = attention_mask
        )
        
        # qAOutput = Softmax(self.lmHead_qAGeneration(qAGenerationDecoderOutput.last_hidden_state), dim = -1)
        qnaLMHeadOutput = self.lmHead_qAGeneration(qAGenerationDecoderOutput.last_hidden_state)

        reconstructionDecoderOutput = self.reconstructionDecoder(
            input_ids = reconstruction_labels,
            encoder_hidden_states = lastHiddenStates,
            encoder_attention_mask = attention_mask
        )

        # reconstructionOutput = Softmax(self.lmHead_reconstruction(reconstructionDecoderOutput.last_hidden_state), dim = -1)
        reconstructionLMHeadOutput = self.lmHead_reconstruction(reconstructionDecoderOutput.last_hidden_state)
        return encoderOutput, reconstructionLMHeadOutput, qnaLMHeadOutput
    
    def load_model(self, path):
        self.encoder = PeftModel.from_pretrained(self.encoder, path)
        self.reconstructionDecoder = PeftModel.from_pretrained(self.reconstructionDecoder, path)
        self.qAGenerationDecoder = PeftModel.from_pretrained(self.qAGenerationDecoder, path)

        self.lmHead_reconstruction.load_state_dict(Load(f"{path}/lmHead_reconstruction", map_location="cuda"))
        self.lmHead_qAGeneration.load_state_dict(Load(f"{path}/lmHead_qAGeneration", map_location="cuda"))
    
    def save_model(self, path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_peft_model(self.encoder, f"{path}/encoder")
        save_peft_model(self.reconstructionDecoder, f"{path}/reconstructionDecoder")
        save_peft_model(self.qAGenerationDecoder, f"{path}/qAGenerationDecoder")
            
        Save(self.lmHead_reconstruction.state_dict(), f"{path}/lmHead_reconstruction")
        Save(self.lmHead_qAGeneration.state_dict(), f"{path}/lmHead_qAGeneration")
