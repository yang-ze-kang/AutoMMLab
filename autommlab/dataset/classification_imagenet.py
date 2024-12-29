import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base import DatasetBase
import numpy as np
import random

from autommlab.configs import DATASET_ZOO

class DatasetImageNetClassification(DatasetBase):

    def __init__(self,
                 data_root = DATASET_ZOO['ImageNet'],
                 root_train_path = 'annos/imagenet/train.txt',
                 root_val_path = 'annos/imagenet/val.txt',
                 root_test_path = '',
                 label_path='annos/imagenet/labels.txt') -> None:
        self.data_root = data_root
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_train_path = os.path.join(file_dir,root_train_path)
        self.root_val_path = os.path.join(file_dir,root_val_path)
        self.label_path = os.path.join(file_dir,label_path)
        labels = []
        label2id = {}
        ds = np.genfromtxt(self.label_path,dtype=str,delimiter='\n')
        for i, d in enumerate(ds):
            for v in d.split(','):
                v=v.strip()
                label2id[v] = i
                labels.append(v)
        self.CLASSES = labels
        self.label2id = label2id

        super().__init__(
            dataset_name='ImageNet1k',
            tag='in1k'
        )
    
    def generate_meta(self,targets, out_dir,sample_num=200):
        if isinstance(targets,list):
            targets = self.search(targets)
        os.makedirs(out_dir, exist_ok=True)
        self.meta_dir = out_dir
        targets_list = list(targets.keys())
        summary = {"target2labels":{}, "target2num":{}}
        id2newId = {}
        for i, tar in enumerate(targets_list):
            summary['target2labels'][tar] = []
            summary['target2num'][tar] = {'train':0,'val':0,'total':0}
            for label in targets[tar]:
                id2newId[self.label2id[label]] = i
                summary['target2labels'][tar].append(label)
        self.metainfo = {
            'classes':list(targets)
        }
        with open(os.path.join(out_dir,'labels.txt'),'w') as f:
            for tar in targets:
                f.write(tar+'\n')
        with open(os.path.join(out_dir,'train.txt'), 'w') as fout:
            data = np.genfromtxt(self.root_train_path,dtype=str,delimiter=' ')
            for d in data:
                id = int(d[1])
                if id in id2newId:
                    fout.write(f"{d[0]} {id2newId[id]}\n")
                    summary['target2num'][targets_list[id2newId[id]]]['train']+=1
                    summary['target2num'][targets_list[id2newId[id]]]['total']+=1
        vals = []
        data = np.genfromtxt(self.root_val_path,dtype=str,delimiter=' ')
        for d in data:
            id = int(d[1])
            if id in id2newId:
                vals.append(d)
        vals = random.sample(vals,min(len(vals),sample_num))
        with open(os.path.join(out_dir,'val.txt'), 'w') as fout:
            for d in vals:
                id = int(d[1])
                if id in id2newId:
                    fout.write(f"{d[0]} {id2newId[id]}\n")
                    summary['target2num'][targets_list[id2newId[id]]]['val']+=1
                    summary['target2num'][targets_list[id2newId[id]]]['total']+=1
        self.train_path = 'train.txt'
        self.val_path = 'val.txt'
        summary['num_classes'] = len(summary['target2labels'])
        summary['dataset'] = self.dataset_name
        summary['tag'] = self.tag
        return summary
    
if __name__=='__main__':
    objects = ['bird']
    dataset = DatasetImageNetClassification()
    tar2label = dataset.search(objects)
    tar2label = {
			"beetles": [
				"tiger beetle",
				"ladybug",
				"ladybeetle",
				"lady beetle",
				"ladybird",
				"ladybird beetle",
				"ground beetle",
				"carabid beetle",
				"long-horned beetle",
				"longicorn",
				"longicorn beetle",
				"leaf beetle",
				"chrysomelid",
				"dung beetle",
				"rhinoceros beetle",
				"weevil"
			],
			"butterflies": [
				"peacock",
				"admiral",
				"ringlet",
				"ringlet butterfly",
				"monarch",
				"monarch butterfly",
				"milkweed butterfly",
				"Danaus plexippus",
				"cabbage butterfly",
				"sulphur butterfly",
				"sulfur butterfly",
				"lycaenid",
				"lycaenid butterfly"
			],
			"bees": [
				"bee"
			]
		}
    path = '/data3/yangzekang/LLMaC/llama_inference/data_test_rectify/classification/bags/logs/req109'
    summary = dataset.generate_meta(tar2label,path)
    import json
    with open(os.path.join(path,'summary.json'),'w') as f:
        json.dump(summary,f,indent='\t')
    print(tar2label)
    print(summary)
    check = dataset.check(objects)
    print(check)
    # if isinstance(check,list):
    #     print(check)
    # else:
    #     print()