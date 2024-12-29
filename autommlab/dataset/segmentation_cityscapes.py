import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base import DatasetBase
import numpy as np
import random
import json

from autommlab.configs import DATASET_ZOO

class DatasetCityscapesSegmentation(DatasetBase):

    CLASSES=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle')
    
    PALETTE=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180],
            [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    
    def __init__(self,
                 config_path = None,
                 crop_size = (512, 1024),
                 data_root=DATASET_ZOO['cityscapes'],
                 root_train_path='annos/cityscapes/train.txt',
                 root_val_path='annos/cityscapes/val.txt',
                 img2subj_path = 'annos/cityscapes/img2subj.json',
                 train_data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                 val_data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                 test_path='',
                 label_path='') -> None:
        self.config_path = config_path
        self.data_root = data_root
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_train_path = os.path.join(file_dir,root_train_path)
        self.root_val_path = os.path.join(file_dir,root_val_path)
        self.img2subj_path = os.path.join(file_dir,img2subj_path)
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        super().__init__(
            dataset_name='Cityscapes',
            tag = 'cityscapes',
            dataset_type='CityscapesDataset'
        )
    
    def generate_meta(self,targets,out_dir=None,sample_num=200):
        if isinstance(targets,list):
            targets = self.search(targets)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        classes = ['background']
        palette = []
        summary = {"target2labels":{}, "label2num":{'total':{'train':0,'val':0,'total':0}}}
        targets_list = list(targets.keys())
        for obj in targets_list:
            summary['target2labels'][obj] = []
            for label in targets[obj]:
                summary['label2num'][label] = {'train':0,'val':0,'total':0}
                summary['target2labels'][obj].append(label)
                classes.append(label)
                palette.append(self.PALETTE[self.CLASSES.index(label)])
        if len(classes)==0:
            return None
        self.metainfo = {
            "classes":classes,
            # "palette":palette
        }
        summary['metainfo'] = self.metainfo

        with open(self.img2subj_path,'r') as f:
            img2subj = json.load(f)
        train_path = os.path.join(out_dir,'train.txt')
        val_path = os.path.join(out_dir,'val.txt')
        with open(train_path,'w') as f:
            lines = np.genfromtxt(self.root_train_path,dtype=str,delimiter='\n')
            for line in lines:
                flag = False
                for cls in classes:
                    if cls in img2subj[line]:
                        flag = True
                        summary['label2num'][cls]['train']+=1
                        summary['label2num'][cls]['total']+=1
                if flag:
                    f.write(f"{line}\n")
                    summary['label2num']['total']['train']+=1
        vals = []        
        lines = np.genfromtxt(self.root_val_path,dtype=str,delimiter='\n')
        for line in lines:
            for cls in classes:
                if cls in img2subj[line]:
                    vals.append(line)
                    break
        vals = random.sample(vals,min(len(vals),sample_num))
        with open(val_path,'w') as f:
            for val in vals:
                for cls in classes:
                    if cls in img2subj[val]:
                        summary['label2num'][cls]['val']+=1
                        summary['label2num'][cls]['total']+=1
                f.write(f"{val}\n")
                summary['label2num']['total']['val']+=1
        summary['label2num']['total']['total']=summary['label2num']['total']['train']+summary['label2num']['total']['val']
        self.train_path = 'train.txt'
        self.val_path = 'val.txt'
        summary['num_classes'] = len(classes)
        summary['dataset'] = 'Cityscapes'
        summary['tag'] = 'cityscapes'
        return summary