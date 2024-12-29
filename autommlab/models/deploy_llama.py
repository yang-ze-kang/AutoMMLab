import torch
import os
import sys
import time
from typing import List
import re
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from peft import PeftModel


from autommlab.configs import PATH_LLAMA2, PATH_LORAS

    
class LLaMA2():

    def __init__(self,model_name=None,
                 peft_model=None,
                 quantization: bool=False,
                 max_new_tokens =4000, #The maximum numbers of tokens to generate
                 prompt_file: str=None,
                 seed: int=42, #seed value for reproducibility
                 do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
                 min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
                 use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
                 top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                 temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
                 top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
                 repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
                 length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
                 enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
                 max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
                 use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
                 ) -> None:
        self.enable_sensitive_topics = enable_sensitive_topics
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.min_length = min_length
        self.use_cache = use_cache
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.max_padding_length = max_padding_length
        
        # Set the seeds for reproducibility
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            load_in_8bit=quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if peft_model:
            model = load_peft_model(model, peft_model)
        model.eval()
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(
            {
            
                "pad_token": "<PAD>",
            }
        )
        self.tokenizer = tokenizer
        model.resize_token_embeddings(model.config.vocab_size + 1)
        model = PeftModel.from_pretrained(model, PATH_LORAS['ru-llama2'], adapter_name="ru-llama2")
        for key in PATH_LORAS:
            if key != 'ru-llama2':
                model.load_adapter(PATH_LORAS[key], adapter_name=key)
        self.model = model
    
    def set_lora(self,name):
        assert name in PATH_LORAS
        self.model.set_adapter(name)

    def __call__(self, prompt, history=None, temperature=0.6):
        self.temperature = temperature
        batch = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_padding_length, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                top_p=self.top_p,
                temperature=self.temperature,
                min_length=self.min_length,
                use_cache=self.use_cache,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        res = output_text.replace(prompt,'')
        res = res.replace(prompt.replace('</s>',' '),'')
        return res





app = Flask(__name__)


@app.route('/llama2', methods=['POST'])
def process_api():
    data = request.get_json()
    peft_model = data.get('peft_model', None)
    model.set_lora(peft_model)
    kwargs = {
        "prompt":data.get('prompt', None),
        "temperature":data.get('temperature', 0.6)
    }
    res = model(**kwargs)
    print(res)
    return res

model = LLaMA2(model_name=PATH_LLAMA2)
if __name__ == '__main__':
    app.run(debug=False,  port=10069)

    