import re
import openai
import json
import numpy as np
import os

from .prompts import ru_init_prompt, ru_example_prefix_prompt, ru_examples_prompt, ru_req_prefix_prompt
from .call_gpt import gpt4, gpt35


def compress_json(data):
    if isinstance(data,str):
        return json.dumps(json.loads(data), separators=(',', ':'))
    elif isinstance(data,dict):
        return json.dumps(data, separators=(',', ':'))
    else:
        raise NotImplementedError

class RUGPT4():

    def __init__(self,example_num=0,iter=1,log_dir='logs') -> None:
        assert example_num<=len(ru_examples_prompt)
        self.iter = iter
        self.example_num = example_num
        self.log_dir = os.path.join(log_dir,f"en{example_num}")
        os.makedirs(self.log_dir,exist_ok=True)

    
    def run(self, query, temperature=0.1):
        prompt = ru_init_prompt
        if self.example_num>0:
            prompt=prompt+'\n'+ru_example_prefix_prompt+'\n'
            for i in range(self.example_num):
                prompt = prompt+ru_examples_prompt[i]+'\n'
        prompt = prompt+'\n'+ru_req_prefix_prompt+query
        message = [
            {"role": "user", "content": prompt},
        ]
        try: 
            response = gpt4(message,temperature=temperature)
            if self.log_dir is not None:
                with open(os.path.join(self.log_dir,f"parse_{self.iter}_t{temperature}.txt"),'w') as f:
                    f.write(str(response)+'\n')
                self.iter+=1
            res = response.choices[0].message.content
            return self.parse_result(res)
        except:
            return 'error'
    
    def parse_result(self,res):
        s = res
        for m in re.findall("###.*###", res):
            s = s.replace(m,"")
        s = s.strip()
        s = s.strip('`')
        if s.startswith("json"):
            s = s[len("json"):]
        s = s.strip()
        data = json.loads(s)
        return data
    
class RUGPT35():

    def __init__(self,example_num=0,iter=1,log_dir='logs') -> None:
        assert example_num<=len(ru_examples_prompt)
        self.iter = iter
        self.example_num = example_num
        self.log_dir = os.path.join(log_dir,f"en{example_num}")
        os.makedirs(self.log_dir,exist_ok=True)

    
    def run(self, query, temperature=0.1):
        prompt = ru_init_prompt
        if self.example_num>0:
            prompt=prompt+'\n'+ru_example_prefix_prompt+'\n'
            for i in range(self.example_num):
                prompt = prompt+ru_examples_prompt[i]+'\n'
        prompt = prompt+'\n'+ru_req_prefix_prompt+query
        message = [
            {"role": "user", "content": prompt},
        ]
        try: 
            response = gpt35(message,temperature=temperature)
            if self.log_dir is not None:
                with open(os.path.join(self.log_dir,f"parse_{self.iter}_t{temperature}.txt"),'w') as f:
                    f.write(str(response)+'\n')
                self.iter+=1
            res = response.choices[0].message.content
            return self.parse_result(res)
        except:
            return 'error'
    
    def parse_result(self,res):
        s = res
        for m in re.findall("###.*###", res):
            s = s.replace(m,"")
        s = s.strip()
        s = s.strip('`')
        if s.startswith("json"):
            s = s[len("json"):]
        s = s.strip()
        data = json.loads(s)
        return data