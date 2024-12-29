import numpy as np
import time
import json
import sys
import os

from autommlab.models.llama2 import LLaMA2
from autommlab.agents.prompts import HPO_INIT_PROMPT


class AgentHPO():

    def __init__(self,model_name='hpo-llama2') -> None:
        self.model_name = model_name
        if model_name=='hpo-llama2':
            self.model = LLaMA2()
        else:
            raise NotImplementedError
    
    def save_summary(self, summary,out_dir):
        with open(os.path.join(out_dir,'summary.json'),'w') as f:
            json.dump(summary, f, indent='\t')

    def __call__(self, req, task):
        prompt = HPO_INIT_PROMPT+"User:"+self.compress_json(req[0]['user'])+'\n'
        i = 0
        while i+1<len(req):
            prompt=prompt+"Assistant:"+self.compress_json(req[i]['assistant'])+'\n</s>'
            prompt=prompt+"User:"+req[i+1]['user']+'\n'
            i+=1
        prompt = prompt+"Assistant:"

        kwargs = {'prompt':prompt,'temperature':0.6}
        if self.model_name=='hpo-llama2':
            kwargs['peft_model'] = 'hpo-llama2'+f'-{task}'
        parse = self.model(**kwargs)
        if isinstance(parse,str):
          parse = eval(parse)
        return parse
    
    def compress_json(self,data):
        if isinstance(data,str):
            return json.dumps(json.loads(data), separators=(',', ':'))
        elif isinstance(data,dict):
            return json.dumps(data, separators=(',', ':'))
        else:
            raise NotImplementedError
        
    