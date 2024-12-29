import json
import os
import statistics

    

def statistic(path):
    metrics_dict = {
        "classification":['accuracy','f1'],
        "detection":['mAP'],
        "segmentation":['mIoU'],
        "pose":['mAP']
    }
    with open(path,'r') as f:
       ds = json.load(f)
    metric2vals = {}
    for d in ds:
        task,_ = d['id'].split('_')
        metircs = metrics_dict[task]
        if d['result'] is None:
            for key in metircs:
                if key not in metric2vals:
                    metric2vals[key] = [0]
                else:
                    metric2vals[key].append(0)
        else:
            for key in metircs:
                if d['result'][key]==-1:
                    val=0
                else:
                    val=d['result'][key]
                if key not in metric2vals:
                    metric2vals[key] = [val]
                else:
                    metric2vals[key].append(val)
    for key in metric2vals:
        print(f"==={key}===")
        print(metric2vals[key])
        # print(sorted(metric2vals[key]))
        print(f"mean:{statistics.mean(metric2vals[key])}")
        print(f"std:{statistics.stdev(metric2vals[key])}")
        print(f"median:{statistics.median(metric2vals[key])}")
    return statistics.mean(metric2vals[key])
   

def increase(data_dir,r1=1,r2=2,target_metric='accuracy'):
    path1 = os.path.join(data_dir,f"val_r{r1}_bag_result.json")
    with open(path1,'r') as f:
        ds = json.load(f)
        id2val = {}
        for d in ds:
            assert d['req_id'] not in id2val
            id2val[d['req_id']] = [d['result'][target_metric]]
    path2 = os.path.join(data_dir,f"val_r{r2}_bag_result.json")
    with open(path2,'r') as f:
        ds = json.load(f)
        for d in ds:
            id2val[d['req_id']].append(d['result'][target_metric])
            print(d['req_id'],id2val[d['req_id']])
    incs = []
    incs_per = []
    for key,val in id2val.items():
        incs.append(val[1]-val[0])
        incs_per.append((val[1]-val[0])/val[0]*100)
    print(f"===increase===")
    print(sorted(incs))
    print(f"mean:{statistics.mean(incs)}")
    print(f"std:{statistics.stdev(incs)}")
    print(f"median:{statistics.median(incs)}")
    print(f"===increase percentage===")
    print(sorted(incs_per))
    print(f"mean:{statistics.mean(incs_per)}")
    print(f"std:{statistics.stdev(incs_per)}")
    print(f"median:{statistics.median(incs_per)}")
    

if __name__=='__main__':
    model = 'llama2-7b-chat'
    root_dir = f'LAMP/results/{model}'
    metrics = {
        'classification':'accuracy',
        'detection':'mAP',
        'segmentation':'mIoU',
        'pose':'mAP'
    }
    # for iter in ['r1','r2','r1-3best']:
    # for iter in ['r1-3best']:
    for iter in ['r1']:
        vals = []
        for task in ['classification','detection','segmentation','pose']:
            path = os.path.join(root_dir,f"results_{task}_{iter}.json")
            print(f"\n==={task}===")
            val = statistic(path)
            vals.append(val)