"""
The model architecture for the Encoder-2*Decoder model with QLoRA.
"""
import torch
from torch import nn as NeuralNetwork, softmax as Softmax, save as Save, load as Load
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from copy import deepcopy
import os

class Model(NeuralNetwork.Module):
    def __init__(self, modelName, inference_mode : bool = False, useLORA : bool = False, bitsAndBytesConfig = None):
        super(Model, self).__init__()

        self.modelName = modelName
        # Load tokenizer
        self.tokenizer = self.get_tokenizer()
        self.tokenizer.sep_token = '[qna-end]\n'
        
        if bitsAndBytesConfig is not None:
            self.quantizationConfig = BitsAndBytesConfig(**bitsAndBytesConfig)
        
        # Load encoder and decoders
        self.model = self.getEncoderDecoders(modelName)
        self.model_dimension = self.model.encoder.config.d_model
        
        if inference_mode:
            self.model.encoder.eval()
            self.model.reconstructionDecoder.eval()
            self.model.qAGenerationDecoder.eval()
            self.model.lmHead_reconstruction.eval()
            self.model.lmHead_qAGeneration.eval()
            return
        else :
            self.model.encoder.train()
            self.model.reconstructionDecoder.train()
            self.model.qAGenerationDecoder.train()
            self.model.lmHead_reconstruction.train()
            self.model.lmHead_qAGeneration.train()

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
        
        totalTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {totalTrainableParams}")
        
        totalModelParamMemory = sum(p.numel() * p.element_size() for p in self.parameters())
        
        # Prepare for k-bit training (e.g., 4-bit)
        self.model.encoder = prepare_model_for_kbit_training(self.model.encoder)
        self.model.reconstructionDecoder = prepare_model_for_kbit_training(self.model.reconstructionDecoder)
        self.model.qAGenerationDecoder = prepare_model_for_kbit_training(self.model.qAGenerationDecoder)
        self.model.lmHead_reconstruction = prepare_model_for_kbit_training(self.model.lmHead_reconstruction)
        self.model.lmHead_qAGeneration = prepare_model_for_kbit_training(self.model.lmHead_qAGeneration)
        
        totalModelParamMemory = sum(p.numel() * p.element_size() for p in self.parameters())
        

        # Configure LoRA for the three modules
        if useLORA:
            self.model = self.configure_lora(self.model)
            for param in self.model.lmHead_reconstruction.parameters():
                param.requires_grad = True
            for param in self.model.lmHead_qAGeneration.parameters():
                param.requires_grad = True
            loRATrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total number of trainable parameters with LoRA: {loRATrainableParams}")
            print(f"Percentage Reduction in Training Params : {(1 - loRATrainableParams/totalTrainableParams) * 100} %")
            

    def getEncoderDecoders(self, modelName):
        if hasattr(AutoConfig, 'quantizationConfig'):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(modelName, quantization_config = self.quantizationConfig)
        else :
            self.model = AutoModelForSeq2SeqLM.from_pretrained(modelName)

        
        self.model.reconstructionDecoder = deepcopy(self.model.decoder)
        self.model.qAGenerationDecoder = deepcopy(self.model.decoder)
        del self.model.decoder
        
        # Randomize the lm_head
        # model.lm_head = NeuralNetwork.Linear(model.config.d_model, model.config.vocab_size)
        
        if not hasattr(self.model, 'lm_head'):
            print(f"No lm_head found in {modelName}. Training the model from scratch.")
            self.model.lm_head = NeuralNetwork.Linear(self.model.config.d_model, self.model.config.vocab_size)

        self.model.lmHead_reconstruction = deepcopy(self.model.lm_head)
        self.model.lmHead_qAGeneration = deepcopy(self.model.lm_head)
        del self.model.lm_head
        
        return self.model
    
    def configure_lora(self, module):
        taskType = 'Seq2SeqLM'
        
        # targetModules = [ moduleName for moduleName, _ in module.named_modules() if moduleName != '' and moduleName.endswith('.k') or moduleName.endswith('.q') ]

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            bias="none",
            lora_dropout=0.01,
            task_type=taskType,
        )
        return get_peft_model(module, peft_config)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.modelName, clean_up_tokenization_spaces = True)

    def forward(self, input_ids, attention_mask, qna_labels, reconstruction_labels):
        encoderOutput = self.model.encoder(input_ids = input_ids, attention_mask = attention_mask)
        lastHiddenStates = encoderOutput.last_hidden_state
        
        # Forward through both decoders
        qAGenerationDecoderOutput = self.model.qAGenerationDecoder(
            input_ids = qna_labels,
            encoder_hidden_states = lastHiddenStates,
            encoder_attention_mask = attention_mask
        )
        
        # qAOutput = Softmax(self.lmHead_qAGeneration(qAGenerationDecoderOutput.last_hidden_state), dim = -1)
        qnaLMHeadOutput = self.model.lmHead_qAGeneration(qAGenerationDecoderOutput.last_hidden_state)

        reconstructionDecoderOutput = self.model.reconstructionDecoder(
            input_ids = reconstruction_labels,
            encoder_hidden_states = lastHiddenStates,
            encoder_attention_mask = attention_mask
        )

        # reconstructionOutput = Softmax(self.lmHead_reconstruction(reconstructionDecoderOutput.last_hidden_state), dim = -1)
        reconstructionLMHeadOutput = self.model.lmHead_reconstruction(reconstructionDecoderOutput.last_hidden_state)
        return encoderOutput, reconstructionLMHeadOutput, qnaLMHeadOutput
    
    def getLogits(self, input_ids, attention_mask, decoder_input_ids):
        """
        Computes the logits for the next token in the sequence.

        Args:
            input_ids (torch.Tensor): Input IDs for the encoder (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for the encoder (batch_size, seq_len).
            decoder_input_ids (torch.Tensor): Current decoder input IDs (batch_size, decoder_seq_len).

        Returns:
            torch.Tensor: Logits for the next token (batch_size, vocab_size).
        """
        # Step 1: Encode the input
        encoder_output = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = encoder_output.last_hidden_state

        # Step 2: Pass the decoder input to the QnA decoder
        decoder_output = self.model.qAGenerationDecoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=last_hidden_states,
            encoder_attention_mask=attention_mask
        )

        # Step 3: Compute the logits for the next token
        logits = self.model.lmHead_qAGeneration(decoder_output.last_hidden_state[:, -1, :])  # Only take the last token's logits
        return logits
    

    def inference(self, input_ids, attention_mask, max_length, sampler, *args, **kwargs):
        """
        Generates a sequence of tokens using the QnA decoder.
        Args:
            input_ids (torch.Tensor): Input IDs for the encoder (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for the encoder (batch_size, seq_len).
            max_length (int): Maximum length of the generated sequence.
            eos_token_id (int): Token ID representing the end of sequence.
        Returns:
            torch.Tensor: Generated sequence of token IDs (batch_size, generated_seq_len).
        """
        # Step 1: Initialize decoder input with the start token
        start_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id or self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        decoder_input_ids = torch.tensor([[start_token_id]]).to(input_ids.device)

        # Step 2: Iteratively generate tokens
        for _ in range(max_length):
            # Get the logits for the next token
            logits = self.getLogits(input_ids, attention_mask, decoder_input_ids)

            # Step 3: Select the token with the highest probability (greedy decoding)
            probabilityDistribution = torch.softmax(logits, dim=-1)
            next_token_id, probability = sampler(probabilityDistribution, *args, **kwargs)
            next_token_id = next_token_id.unsqueeze(0)

            # Step 4: Append the generated token to the decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)

            # Step 5: Check if the EOS token has been generated
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            yield decoder_input_ids
        
        return decoder_input_ids

    def load_model(self, path):
        self.model = PeftModel.from_pretrained(self.model, os.path.abspath(path))

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        print(f"Saving model to {path}")
        self.model.save_pretrained(path, save_adapter=True, save_config=True)
        print("Model saved at ", path)
