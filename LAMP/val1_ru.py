import os
import json
import re
from thefuzz import fuzz
from agents import RUGPT4, RUGPT35, RULLaMA, RULLaMAChat
import time
import torch

@torch.no_grad()
def run(model, path, out_path):
    if model=='llama2-7b-chat':
        llm = RULLaMAChat(example_num=1, log_dir=f'logs/{model}')
    if model=='llama2':
        llm = RULLaMA()
    elif model=='gpt4':
        llm = RUGPT4(example_num=1, log_dir='logs/gpt4')
    elif model=='gpt3.5':
        llm = RUGPT35(example_num=1, log_dir='logs/gpt3.5')
    start_flag = True
    last_time = time.time()
    with open(path,'r') as f, open(out_path.replace('.json','_raw.json'),'a') as f2:
        f2.write('[\n')
        ds = json.load(f)
        for i,d in enumerate(ds):
            o = llm.run(d['requirement'])
            d['parse'] = o
            now_time = time.time()
            t = now_time - last_time
            last_time = now_time
            if not start_flag:
                f2.write(',\n')
            json.dump(d,f2,indent='\t')
            start_flag = False
            print(f"{i+1}/{len(ds)},{t}s")
            f2.flush()
        f2.write('\n]')

def post_process(path):
    with open(path.replace('.json','_raw.json'),'r') as f:
        ds = json.load(f)
        for d in ds:
            d['parse'] = json.loads(d['parse'])
    with open(path,'w') as f:
        json.dump(ds,f,indent='\t')
def find_brackets_positions(text):
    bracket_positions = []
    stack = []
    for match in re.finditer(r'[{}]', text):
        char = match.group()
        if char == '{':
            stack.append(match.start())
        elif char == '}':
            if stack:
                start_pos = stack.pop()
                bracket_positions.append((start_pos, match.start()))
            else:
                print("Error: Mismatched closing bracket '}' at position", match.start())
    if stack:
        print("Error: Mismatched opening bracket '{' at position", stack.pop())
    return bracket_positions

def post_process_log(log_dir,path,out_path):
    with open(path,'r') as f:
        ds = json.load(f)
    res = []
    for i,file in enumerate(os.listdir(log_dir)):
        with open(os.path.join(log_dir,file),'r') as f:
            res = f.read()
            poss = find_brackets_positions(res)
            mini = len(res)
            target = None
            for pos in poss:
                if pos[0] < mini:
                    target = pos
                    mini = pos[0]
            res = res[target[0]:target[1]+1]
            try:
                ds[i]['parse'] = json.loads(res)
            except:
                print(res)
                print(poss)
                print(target)
                return
    with open(out_path,'w') as f:
        json.dump(ds,f,indent='\t')
    
def are_same(word1, word2):
    word1 = word1.lower()
    word2 = word2.lower()
    word1 = word1.replace(' ','_')
    word2 = word2.replace(' ','_')
    return fuzz.partial_ratio(word1,word2)>=95

def get_metirc(name):
    name = name.lower()
    name = name.replace('rank','top')
    pattern = r'ap\s*at\s*iou\s*=?\s*(\d+\.\d+)'
    replacement = r'AP@\1'
    name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    if name in ['localization accuracy','classification accuracy','detection accuracy']:
        return 'accuracy'
    elif name in ['tracking precision','precision']:
        return 'precision'
    elif name in ['ap', 'detection ap', 'average precision','map']:
        return 'AP'
    elif name in ['tpr','sensitivity','recall']:
        return 'TPR'
    elif name in ['iou','jaccard index','miou', 'mean iou']:
        return 'IoU'
    elif name in ['dice coefficient','dice score','dice similarity coefficient','dsc']:
        return 'Dice coefficient'
    elif name in ['f1','f1 score','f1-score']:
        return 'F1'
    elif name in ['pixel accuracy','pixel-wise accuracy']:
        return 'pixel accuracy'
    elif 'map@' in name:
        name = name.replace('map@','mAP@')
    elif 'pk' in name:
        return False
    elif 'pck' in name:
        name = name.replace(' ','')
        name = name.replace('pck','PCK')
    elif name in ['auc','fpr','mae']:
        name = name.upper()
    return name



def summary_score(test_path,gt_path):
    with open(gt_path,'r') as f:
        ds = json.load(f)
        req2gt = {}
        for d in ds:
            assert d['id'] not in req2gt
            req2gt[d['id']] = d['parse']
    with open(test_path,'r') as f:
        ds = json.load(f)
        req2test = {}
        for d in ds:
            assert d['id'] not in req2test
            req2test[d['id']] = d['parse']
    assert len(req2gt)==len(req2test)
    assert set(req2gt.keys())==set(req2test.keys())
    keys = ['format','data_object','model_task','model_specific','model_speed','model_flops','model_params','model_metric','deploy_gpu','deploy_engine']
    key2error_id = {key:[] for key in keys}
    ids = set(req2test.keys())
    for id in ids:
        print(id)
        # formate error
        if req2test[id]=='error' or \
            'data' not in req2test[id] or 'object' not in req2test[id]['data'] or \
            'model' not in req2test[id] or 'task' not in req2test[id]['model']:
            key2error_id['format'].append(id)
            continue

        ## data ##
        gt = req2gt[id]['data']
        pred = req2test[id]['data']

        # data objects
        key = 'data_object'
        if len(pred['object'])!=len(gt['object']):
            key2error_id[key].append(id)
        else:
            l1 = sorted(pred['object'])
            l2 = sorted(gt['object'])
            for i in range(len(l1)):
                if not are_same(l1[i],l2[i]):
                    key2error_id[key].append(id)
                    break

        ## model ##
        gt = req2gt[id]['model']
        pred = req2test[id]['model']

        # model task
        key = 'model_task'
        if gt['task']!=pred['task']:
            key2error_id[key].append(id)

        # model specific_model
        key = 'model_specific'
        if pred['specific_model']=='not specified':
            pred['specific_model'] = 'none'
        if gt['specific_model'].lower()!=pred['specific_model'].lower():
            key2error_id[key].append(id)

        # model speed
        key = 'model_speed'
        for k in ['value','unit']:
            if pred['speed'][k]!=gt['speed'][k]:
                key2error_id[key].append(id)
                break

        # model flops
        key = 'model_flops'
        for k in ['value','unit']:
            if pred['flops'][k]!=gt['flops'][k]:
                key2error_id[key].append(id)
                break

        # model parameters
        key = 'model_params'
        for k in ['value','unit']:
            if pred['parameters'][k]!=gt['parameters'][k]:
                key2error_id[key].append(id)
                break

        # model metric
        key = 'model_metric'
        if len(pred['metrics'])!=len(gt['metrics']):
            key2error_id[key].append(id)
        elif len(pred['metrics'])!=0 and len(gt['metrics'])!=0:
            l1 = sorted(pred['metrics'],key=lambda x:get_metirc(x['name']))
            l2 = sorted(gt['metrics'],key=lambda x:get_metirc(x['name']))
            for a,b in zip(l1,l2):
                if get_metirc(a['name'])!=get_metirc(b['name']):
                    key2error_id[key].append(id)
                    break
                elif a['value']!=b['value']:
                    key2error_id[key].append(id)
                    break

        ## deploy ##
        gt = req2gt[id]['deploy']
        pred = req2test[id]['deploy']
        # deploy gpu
        key = 'deploy_gpu'
        if pred['gpu']!=gt['gpu']:
            key2error_id[key].append(id)
        # deploy engine
        key = 'deploy_engine'
        if pred['inference engine']!=gt['inference engine']:
            key2error_id[key].append(id)
    """
    summary
    """
    for key in key2error_id:
        key2error_id[key] = sorted(key2error_id[key])
    print(key2error_id)
    # item
    item_keys = ['model_task','model_specific','model_speed','model_flops','model_params','deploy_gpu','deploy_engine']
    ids = []
    for key in item_keys:
        ids.extend(key2error_id[key])
    ids = list(set(ids))
    item_error_num = len(ids) + len(key2error_id['format']) * len(item_keys)
    item_num = len(item_keys)*80
    print(f"item-level:{item_error_num}/{item_num},{(item_num-item_error_num)/item_num*100}%")
    list_keys = ['data_object','model_metric']
    ids = []
    for key in list_keys:
        ids.extend(key2error_id[key])
    ids = list(set(ids))
    list_error_num = len(ids) + len(key2error_id['format']) * len(list_keys)
    list_num = len(list_keys)*80
    print(f"list-level:{list_error_num}/{list_num},{(list_num-list_error_num)/list_num*100}%")
    print(f"key-level:{item_error_num+list_error_num}/{item_num+list_num},{(item_num+list_num-item_error_num-list_error_num)/(item_num+list_num)*100}%")
    ids = []
    for key in keys:
        ids.extend(key2error_id[key])
    ids = list(set(ids))
    req_error_num = len(ids)
    print(f"req-level:{req_error_num}/80,{(80-req_error_num)/80*100}%")


if __name__=='__main__':
    model = 'llama2-7b-chat'
    path = 'data/test_data/annotation.json'
    out_path = f'results/{model}/req&parse.json'
    os.makedirs(f'results/{model}',exist_ok=True)
    
    run(model,path,out_path)
    
    # post_process(out_path)
    # post_process_log(f"logs/{model}/en1",path,out_path)

    summary_score(out_path,path)
