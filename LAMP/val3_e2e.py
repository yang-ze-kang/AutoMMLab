import json
from thefuzz import fuzz
import numpy as np
import re
import os

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
        return 'f1-score'
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

def test_parse(parse_gt,parse_pred):
    res = {'error':[]}
    # formate error
    if parse_pred=='error' or \
        'data' not in parse_pred or 'object' not in parse_pred['data'] or \
        'model' not in parse_pred or 'task' not in parse_pred['model']:
        res['error'].append('format')
        return res

    ## data ##
    gt = parse_gt['data']
    pred = parse_pred['data']

    # data object
    key = 'data_object'
    if len(pred['object'])!=len(gt['object']):
        res['error'].append(key)
    else:
        l1 = sorted(pred['object'])
        l2 = sorted(gt['object'])
        for i in range(len(l1)):
            if not are_same(l1[i],l2[i]):
                res['error'].append(key)
                break

    ## model ##
    gt = parse_gt['model']
    pred = parse_pred['model']

    # model task
    key = 'model_task'
    if gt['task']!=pred['task']:
        res['error'].append(key)

    # model specific_model
    key = 'model_specific'
    if pred['specific_model']=='not specified':
        pred['specific_model'] = 'none'
    if gt['specific_model'].lower()!=pred['specific_model'].lower():
        res['error'].append(key)

    # model speed
    key = 'model_speed'
    for k in ['value','unit']:
        if pred['speed'][k]!=gt['speed'][k]:
            res['error'].append(key)
            break

    # model flops
    key = 'model_flops'
    for k in ['value','unit']:
        if pred['flops'][k]!=gt['flops'][k]:
            res['error'].append(key)
            break

    # model parameters
    key = 'model_params'
    for k in ['value','unit']:
        if pred['parameters'][k]!=gt['parameters'][k]:
            res['error'].append(key)
            break

    # model metric
    key = 'model_metric'
    if len(pred['metrics'])!=len(gt['metrics']):
        res['error'].append(key)
    elif len(pred['metrics'])!=0 and len(gt['metrics'])!=0:
        l1 = sorted(pred['metrics'],key=lambda x:get_metirc(x['name']))
        l2 = sorted(gt['metrics'],key=lambda x:get_metirc(x['name']))
        for a,b in zip(l1,l2):
            if get_metirc(a['name'])!=get_metirc(b['name']):
                res['error'].append(key)
                break
            elif a['value']!=b['value']:
                res['error'].append(key)
                break

    ## deploy ##
    gt = parse_gt['deploy']
    pred = parse_pred['deploy']
    # deploy gpu
    key = 'deploy_gpu'
    if pred['gpu']!=gt['gpu']:
        res['error'].append(key)
    # deploy engine
    key = 'deploy_engine'
    if pred['inference engine']!=gt['inference engine']:
        res['error'].append(key)
    return res

def summary_res(task,res_paths,out_path):
    metrics_dict = {
        'classification':['accuracy','f1'],
        'detection':['mAP'],
        'segmentation':['mIoU'],
        'pose':['mAP']
    }
    metrics = metrics_dict[task]
    req2res = {}
    for model,path in res_paths.items():
        with open(path,'r') as f:
            ds = json.load(f)
            for d in ds:
                if d['id'] not in req2res:
                    req2res[d['id']] = {}
                req2res[d['id']][model] = d['result']
    models = res_paths.keys()
    for metric in metrics:
        print(f"\n==={metric}===")
        m2vals = {m:[] for m in models}
        for req,val in req2res.items():
            for m,res in val.items():
                m2vals[m].append(res[metric])
        index = np.argsort(m2vals['llama2'])
        ids = list(req2res.keys())
        # print(np.array(ids)[index].tolist())
        print(f"id {list(models)}")
        for i in index:
            print(f"{ids[i]}",end=':')
            for m in models:
                print(m2vals[m][i],end=' ')
            print()

        


def compute(task,parse_gt_path,parse_pred_path,res_path):
    score = 0
    log = {'parse_error':[],'metric_error':[]}
    metrics_dict = {
        'classification':['accuracy','f1'],
        'detection':['mAP'],
        'segmentation':['mIoU'],
        'pose':['mAP']
    }
    metrics = metrics_dict[task]
    gts = {}
    with open(parse_gt_path,'r') as f:
        ds = json.load(f)
        for d in ds:
            cls,id = d['id'].split('_')
            if cls==task:
                assert id not in gts
                gts[id] = d['parse']
    assert len(gts.keys())==20
    # test parse
    with open(parse_pred_path,'r') as f:
        ds = json.load(f)
        req2parse = {}
        for d in ds:
            cls,id = d['id'].split('_')
            if cls==task:
                assert id not in req2parse
                req2parse[id] = d['parse']
    assert len(req2parse)==len(gts)
    assert set(req2parse.keys())==set(gts.keys())
    keys = ['data_object','model_task','model_specific','model_speed','model_flops','model_params','model_metric','deploy_gpu','deploy_engine']
    parse_success_ids = []
    for id in req2parse.keys():
        error = test_parse(gts[id],req2parse[id])
        if len(error['error'])!=0:
            log['parse_error'].append(id)
            continue
        else:
            score+=1
        parse_success_ids.append(id)
    id2metric_pred = {}
    with open(res_path,'r') as f:
        ds = json.load(f)
        for d in ds:
            cls,id = d['id'].split('_')
            if cls==task:
                assert id not in id2metric_pred
                id2metric_pred[id] = d['result']
    # test metric
    for id in parse_success_ids:
        metric_dict = {}
        for metric in gts[id]['model']['metrics']:
            metric_dict[metric['name']] = metric['value']
        flag = True
        for metric in metrics:
            if get_metirc(metric) in metric_dict:
                if metric=='mIoU':
                    id2metric_pred[id][metric]/=100
                if id2metric_pred[id][metric] < metric_dict[get_metirc(metric)]:
                    flag=False
                    log['metric_error'].append(id)
                    break
        if flag:
            score+=1
    return score,log

if __name__=='__main__':
    model = 'llama2-7b-chat'
    # model = 'gpt3-5'
    # model='llama2'
    parse_gt_path = 'LAMP/data/test_data/annotation.json'
    root_dir = f'LAMP/results/{model}'
    parse_pred_paths = {
        'llama2':'',
        'gpt3-5':'results/reqparse_test_gpt3-5-1shot.json',
        'gpt4':'results/reqparse_test_gpt4-1shot.json',
        'palm':'results/reqparse_test_palm2-bison-1shot.json'
    }
    
    total = 0
    for task in ['classification','detection','segmentation','pose']:
        parse_pred_path = os.path.join(root_dir,'req&parse.json')
        res_path = os.path.join(root_dir,f"results_{task}_r1.json")
        # res_path = os.path.join(root_dir,f"results_{task}_r1-3best.json")
        score,logs = compute(task,parse_gt_path,parse_pred_path,res_path)
        print(f'\n{model} {task}\nscore:',score)
        print(logs)
        total+=score
    print(f'total:{total}')

    # out_path = f'{task}/res_summary.json'
    # summary_res(task,res_paths,out_path)