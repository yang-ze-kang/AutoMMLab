import numpy as np
import time
import json
import sys
import os

from autommlab.models.llama2 import LLaMA2
from autommlab.agents.prompts import RU_INIT_PROMPT


class AgentRU():

    task2metric = {
        'classification':'accuracy',
        'detection':'AP',
        'segmentation':'IoU',
        'keypoint':'AP'
    }

    def __init__(self,model_name='ru-llama2') -> None:
        self.model_name = model_name
        if model_name=='ru-llama2':
            self.model = LLaMA2()
        else:
            raise NotImplementedError
    
    def save_summary(self, summary,out_dir):
        with open(os.path.join(out_dir,'summary.json'),'w') as f:
            json.dump(summary, f, indent='\t')

    def __call__(self, input):
        prompt = RU_INIT_PROMPT+input+"\n###parse###"
        kwargs = {'prompt':prompt}
        if self.model_name=='ru-llama2':
            kwargs['peft_model'] = 'ru-llama2'
            kwargs['temperature'] = 0.6
        parse = self.model(**kwargs)
        if isinstance(parse,str):
          parse = eval(parse)
        if 'error' in parse:
            return parse
        parse = self.post_process_parse(parse)
        return parse
        
    
    def post_process_parse(self,parse):
        if parse['model']['task'] == 'keypoint':
            parse['model']['task'] = 'pose'
        if parse['model']['specific_model'] in ['not specific']:
            parse['model']['specific_model'] = 'none' 

        for metric in parse['model']['metrics']:
            if metric['value']==0 or metric['name']=='none':
                metric['name']='none'
                metric['value']=0
            else:
                if metric['name'].lower() in ['tpr','sensitivity','recall']:
                    metric['name'] = 'recall'
                elif metric['name'].lower() in ['precision']:
                    metric['name'] = 'precision'
                elif metric['name'].lower() in ['f1','f1 score','f1-score']:
                    metric['name'] = 'f1-score'
                elif metric['name'].lower() in ['accuracy']:
                    metric['name'] = 'accuracy'
        
        # unities unit
        if parse['model']['parameters']['value'] != 0:
            if parse['model']['parameters']['unit'] == 'B':
                parse['model']['parameters']['value']*=1000
            elif parse['model']['parameters']['unit'] == 'K':
                parse['model']['parameters']['value']/=1000
            elif parse['model']['parameters']['unit'] != 'M':
                parse['model']['parameters']['unit'] = 'none'
            parse['model']['parameters']['unit'] = 'M'
        if parse['model']['flops']['value'] != 0:
            if parse['model']['flops']['unit'] == 'FLOPs':
                parse['model']['flops']['value']/=1e6
            elif parse['model']['flops']['unit'] == 'MFLOPs':
                parse['model']['flops']['value']/=1e3
            elif parse['model']['flops']['unit'] == 'TFLOPs':
                parse['model']['flops']['value']*=1e3
            elif parse['model']['flops']['unit'] == 'PFLOPs':
                parse['model']['flops']['value']*=1e6
            elif parse['model']['flops']['unit'] == 'EFLOPs':
                parse['model']['flops']['value']*=1e9
            elif parse['model']['flops']['unit'] != 'GFLOPs':
                parse['model']['flops']['unit'] = 'none'
            parse['model']['flops']['unit'] = 'GFLOPs'
        if parse['model']['speed']['value'] != 0:
            if parse['model']['speed']['unit'] == 'ms':
                parse['model']['speed']['value'] = 1000.0/parse['model']['speed']['value']
            elif parse['model']['speed']['unit'] == 's':
                parse['model']['speed']['value'] = 1.0/parse['model']['speed']['value']
            elif parse['model']['speed']['unit'] == 'min':
                parse['model']['speed']['value'] = 1/(parse['model']['speed']['value']*60)
            elif parse['model']['speed']['unit'] == 'h':
                parse['model']['speed']['value'] = 1.0/(parse['model']['speed']['value']*3600)
            elif parse['model']['speed']['unit'] == 'fpm':
                parse['model']['speed']['value'] = parse['model']['speed']['value']/60
            elif parse['model']['speed']['unit'] != 'fps':
                parse['model']['speed']['unit'] = 'none'
            parse['model']['speed']['unit'] = 'fps'
        return parse