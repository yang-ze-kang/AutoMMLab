import torch
import time
import json
from transformers import LlamaTokenizer,LlamaForCausalLM, LlamaConfig
from peft import PeftModel
import numpy as np
import random
from .prompts import init_prompt, example_prefix_prompt, examples_prompt, req_prefix_prompt

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

class LLaMa2():

    def __init__(self,model_name='/data3/yangzekang/LLMaC/llama_inference/llama_weights/llama-2-7b-hf',
                 peft_model='/data3/yangzekang/LLMaC/lora_weights',
                 quantization: bool=False,
                 max_new_tokens =3000, #The maximum numbers of tokens to generate
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
                 enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
                 enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
                 enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
                 max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
                 ) -> None:
        self.prompt_prefix = init_prompt
        self.enable_azure_content_safety = enable_azure_content_safety
        self.enable_sensitive_topics = enable_sensitive_topics
        self.enable_salesforce_content_safety = enable_salesforce_content_safety
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
        
        model = load_model(model_name, quantization)
        if peft_model:
            model = load_peft_model(model, peft_model)
        model.eval()

        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(
            {
            
                "pad_token": "<PAD>",
            }
        )
        model.resize_token_embeddings(model.config.vocab_size + 1) 
        self.tokenizer = tokenizer
        self.model = model

    def run(self, req, temperature=0.1):
        self.temperature = temperature
        prompt = init_prompt+req_prefix_prompt+req+"\n###parse###"
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
        # print(f"the inference time is {e2e_inference_time} ms")
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        res = output_text.replace(prompt,'')
        try:
            res = self.parse_result(res)
        except json.decoder.JSONDecodeError:
            print(res)
            return {'error':'parse error!'}
        return res
    
    def decompress_json(self,data):
        if isinstance(data,str):
            return json.dumps(json.loads(data), indent='  ')
        elif isinstance(data,dict):
            return json.dumps(data, indent='\t')
        else:
            raise NotImplementedError
    
    def parse_result(self,res):
        s = self.decompress_json(res.strip())
        return s

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(path,lora_path):
    llm = LLaMa2(
        model_name='/mnt/lustre/share_data/zengwang/llama/llama-2-7b-hf',
        peft_model=lora_path
    )
    out_path = path.replace('.json','_llama2.json')
    with open(path,'r') as f:
        ds = json.load(f)
        for d in ds:
            o = llm.run(d['requirement'])
            print(o)
            d['parse'] = o
    with open(out_path,'w') as f:
        json.dump(ds,f,indent='\t')

def RULLaMA(model_name='/mnt/lustre/share_data/zengwang/llama/llama-2-7b-hf',
        peft_model='/mnt/cachenew/yangzekang/LLMaC/llama-recipes/save_dir_reqparse_v2'):
    llm = LLaMa2(
        model_name=model_name,
        peft_model=peft_model
    )
    return llm

if __name__=='__main__':
    set_seed(0)
    lora_path = '/mnt/cachenew/yangzekang/LLMaC/llama-recipes/save_dir_reqparse_v2'
    path = '/mnt/cachenew/yangzekang/LLMaC/llama_inference/reqparse_test/data/reqparse_test.json'
    main(path,lora_path)
