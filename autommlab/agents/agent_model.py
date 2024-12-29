from mmengine.config import Config
from pathlib import Path
from prettytable import PrettyTable
import numpy as np
from thefuzz import process
import json
import os

from autommlab.utils.util import array_to_markdown_table


PAPERS_ROOT = Path('papers') 

process.extractBests

class AgentModel():

    model_zoos = {
        "classification":"autommlab/model_zoo/classification_model_zoo.json",
        "detection":"autommlab/model_zoo/detection_model_zoo.json",
        "segmentation":"autommlab/model_zoo/segmentation_model_zoo.json",
        "pose_person":"autommlab/model_zoo/pose_person_model_zoo.json",
        'pose_animal':"autommlab/model_zoo/pose_animal_model_zoo.json"
    }

    model_metric = {
        "classification":"Accuracy",
        "detection":"box AP",
        "segmentation":"mIoU",
        "pose_person":"AP",
        "pose_animal":"AP"
    }

    def __init__(self,
                 detection_model_zoo_path = 'autommlab/modelsdetection_model_zoo.json') -> None:
        self.model_index_classify = 'model-index_classification.yml'
        self.detection_model_zoo_path = detection_model_zoo_path

    def run_single(self, parse, out_dir, data_tag=None,mode='hard'):
        self.mode = parse['task']
        task = parse['task']
        if task=='pose':
            if data_tag in ['coco']:
                task = 'pose_person'
            elif data_tag in ['ap10k']:
                task = 'pose_animal'
            else:
                raise NotImplementedError
        table_head, model_table = self.read_model_zoo(self.model_zoos[task])
        head = ["Model", "Params(M)", "Flops(G)", "Inf time (fps)", self.model_metric[task], "Config", "Weights"] 
        assert head == table_head
        head.append('Score')
        if parse['specific_model'] != 'none':
            res = process.extract(parse['specific_model'],model_table[:,0],limit=30)
            indexs,scores = [],[]
            for name,score in res:
                index = np.where(model_table[:,0]==name)[0]
                assert len(index)==1
                index = int(index)
                indexs.append(index)
                scores.append([score])
            model_table = model_table[indexs]
            model_table = np.hstack([model_table, scores])
        else:
            scores = np.zeros((len(model_table),1))
            model_table = np.hstack([model_table,scores])
        if parse['parameters']['value'] != 0:
            tar_params = parse['parameters']['value']
            indexs = []
            if mode=='hard':
                for i, model in enumerate(model_table):
                    if model[1]<=tar_params:
                        indexs.append(i)
            elif mode=='easy':
                while len(indexs)<5:
                    for i, model in enumerate(model_table):
                        if model[1]<=tar_params:
                            indexs.append(i)
                    tar_params*=1.2
            if len(indexs)==0:
                return {"error":f"There is no {task} model with parameters less than {tar_params}M in the model library!"}, ""
            model_table = model_table[indexs]
        if parse['flops']['value'] != 0:
            tar_flops = parse['flops']['value']
            indexs = []
            if mode=='hard':
                for i, model in enumerate(model_table):
                    if model[2]<=tar_flops:
                        indexs.append(i)
            elif mode=='easy':
                while len(indexs)<5:
                    for i, model in enumerate(model_table):
                        if model[2]<=tar_flops:
                            indexs.append(i)
                    tar_flops*=1.2
            if len(indexs)==0:
                return {"error":f"There is no {task} model with Flops less than {tar_flops}G in the model library!"}, ""
            model_table = model_table[indexs]
        if parse['speed']['value'] != 0:
            tar_fps = parse['speed']['value']
            indexs = []
            while len(indexs)<10:
                for i, model in enumerate(model_table):
                    if model[3]>=tar_fps:
                        indexs.append(i)
                tar_fps*=0.75
            model_table = model_table[indexs]
        model_table = sorted(model_table, key=lambda x:(-x[-1], -float(x[-4])))
        model_table = np.array(model_table)
        # print(model_table)
        model_row = model_table[0][:-1]
        model = {key:val for key,val in zip(head,model_row)}
        
        config = Config.fromfile(model['Config'])
        config.dump(os.path.join(out_dir,'config_raw.py'))
        
        return model, self.format_print(head[:5],model_row)
    
    def format_print(self, head, row):
        s = "The following model has been selected based on your needs:\n"
        rows = []
        rows.append(head)
        rows.append(row)
        return s+array_to_markdown_table(rows)
    
    def read_model_zoo(self,path):
        with open(path,'r') as f:
            data = json.load(f)
        head = data['keys']
        models = []
        for d in data['models']:
            model = []
            for h in head:
                model.append(d[h])
            models.append(model)
        models = np.array(models,dtype=object)
        return head, models
    
    def print_model_table(self,head=['model','params','flops','top-1','top-5','config'], table=None, mode='prettytable'):
        if mode == 'prettytable':
            t = PrettyTable(head)
            t.add_rows(table)
            print(t)


if __name__=='__main__':
    agent = AgentModel()
    # table = agent.read_model_table(agent.model_index_classify,'Image Classification')
    # agent.print_model_table(table)
    res = agent.run_single({
    "description": "A model that can accurately classify Huskies and Alaskan malamutes with an accuracy rate above 90%.",
    "task": "detection",
    "specific_model": "Faster-rcnn",
    "speed": {
      "value": 25,
      "unit": "none"
    },
    "flops": {
      "value": 0,
      "unit": "none"
    },
    "parameters": {
      "value": 50,
      "unit": "M"
    },
    "metrics": [
      {
        "name": "mAP",
        "value": 0.85
      }
    ]
  },'')
    print(res)