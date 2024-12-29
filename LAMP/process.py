import os
import json
import shutil
import numpy as np


def get_bags_and_anno():
    bags = {
        "classification":"/data3/yangzekang/LLMaC/llama_inference/bags/classification/bags_v3/bags120",
        "detection":"/data3/yangzekang/LLMaC/llama_inference/bags/detection/v2/bags120",
        "segmentation":"/data3/yangzekang/LLMaC/llama_inference/bags/segmentation/v1/bags120",
        "pose":"/data3/yangzekang/LLMaC/llama_inference/bags/pose/v1/bags120"
    }
    val_paths = {
        "classification":"/data3/yangzekang/LLMaC/llama_inference/data_hp/classification/val.txt",
        "detection":"/data3/yangzekang/LLMaC/llama_inference/data_hp/detection/val.txt",
        "segmentation":"/data3/yangzekang/LLMaC/llama_inference/data_hp/segmentation/val.txt",
        "pose":"/data3/yangzekang/LLMaC/llama_inference/data_hp/pose/val.txt"
    }
    annos = []
    for task in tasks:
        ids = np.genfromtxt(val_paths[task],delimiter='\n',dtype=int)
        for i,id in enumerate(ids):
            name = f"{task}_{i+1}"
            bag_dir = os.path.join(bags[task],f'req{id}')
            save_dir = os.path.join(root_dir,'bags',name)
            with open(os.path.join(bag_dir,'req&parse.json'),'r') as f:
                ds = json.load(f)
                anno = {
                    "id":name,
                    "requirement":ds['requirement'],
                    "parse":ds['parse']
                }
                annos.append(anno)
            shutil.copytree(bag_dir,save_dir,dirs_exist_ok=True)
    with open(os.path.join(root_dir,'annotation_raw.json'),'w') as f:
        json.dump(annos,f,indent='\t')

def results_old2new():
    model = 'llama2-7b-chat'
    val_paths = {
        "classification":"/data3/yangzekang/LLMaC/llama_inference/data_hp/classification/val.txt",
        "detection":"/data3/yangzekang/LLMaC/llama_inference/data_hp/detection/val.txt",
        "segmentation":"/data3/yangzekang/LLMaC/llama_inference/data_hp/segmentation/val.txt",
        "pose":"/data3/yangzekang/LLMaC/llama_inference/data_hp/pose/val.txt"
    }
    for task in ['classification','detection','segmentation','pose']:
    # for task in ['classification']:
    # for task in ['pose']:
        ids = np.genfromtxt(val_paths[task],delimiter='\n',dtype=int)
        res_paths = {
            # 'llama2':f"/data3/yangzekang/LLMaC/llama_inference/data_hp/{task}/val_v2/val_r1_bag_result.json",
            'llama2':f"/data3/yangzekang/LLMaC/llama_inference/data_hp/{task}/val_v2/",
            'gpt3-5':f"/data3/yangzekang/LLMaC/llama_inference/data_hp/{task}/val_gpt3-5/",
            'gpt4':f"/data3/yangzekang/LLMaC/llama_inference/data_hp/{task}/val_gpt4/",
            'palm2':f"/data3/yangzekang/LLMaC/llama_inference/data_hp/{task}/val_palm_t0.2/",
            "llama2-7b-chat":f"/data3/yangzekang/LLMaC/llama_inference/data_hp/{task}/val_llama2-7b-chat/"
        }
        # for iter in ['r1','r2','r3','r1-3best']:
        for iter in ['r1']:
            hp_path = os.path.join(res_paths[model],f"val_{iter}b.json")
            res_path = os.path.join(res_paths[model],f"val_{iter}_bag_result.json")
            id2anno = {}
            with open(res_path,'r') as f:
                ds = json.load(f)
                for d in ds:
                    id2anno[d['req_id']] = {
                        'data':d['data'],
                        'model':d['model'],
                        'hp':d['hp'],
                        'result':d['result']
                    }
            id2anno2 = {}
            with open(hp_path,'r') as f:
                ds = json.load(f)
                for d in ds:
                    id2anno2[d['req_id']] = {
                        'data':d['conversation'][-1]['user']['data'],
                        'model':d['conversation'][-1]['user']['model'],
                        'hp':{'error':'format error'},
                        'result':None
                    }
            res = []
            for i, id in enumerate(ids):
                if id in id2anno:
                    d = id2anno[id]
                else:
                    d = id2anno2[id]
                anno = {
                    'id':f"{task}_{i+1}",
                    'data':d['data'],
                    'model':d['model'],
                    'hp':d['hp'],
                    'result':d['result']
                }
                res.append(anno)
            with open(os.path.join(results_dir,model,f'results_{task}_{iter}.json'),'w') as f:
                json.dump(res,f,indent='\t')
        

if __name__=='__main__':
    root_dir = '/data3/yangzekang/LLMaC/LAMP/data/test_data'
    results_dir = '/data3/yangzekang/LLMaC/LAMP/results'
    tasks = ['classification','detection','segmentation','pose']
    # get_bags_and_anno()

    results_old2new()
