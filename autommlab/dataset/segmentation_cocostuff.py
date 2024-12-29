import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base import DatasetBase
import numpy as np
import random
import json

class DatasetCocostuffSegmentation(DatasetBase):

    CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood')
    
    PALETTE=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                 [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                 [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                 [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                 [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                 [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                 [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                 [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                 [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                 [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                 [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                 [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                 [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                 [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                 [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                 [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                 [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                 [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                 [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                 [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                 [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                 [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                 [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                 [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                 [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                 [64, 192, 96], [64, 160, 64], [64, 64, 0]]
    
    def __init__(self,
                 config_path = None,
                 crop_size = (512, 512),
                 scale = (2048, 512),
                 data_root='sh1984:s3://openmmlab/datasets/segmentation/coco_stuff164k',
                 root_train_path='annos/coco_stuff164k/train.txt',
                 root_val_path='annos/coco_stuff164k/val.txt',
                 img2subj_path = 'annos/coco_stuff164k/img2subj.json',
                 train_data_prefix=dict(img_path='images/train2017', seg_map_path='annotations/train2017'),
                 val_data_prefix = dict(img_path='images/val2017', seg_map_path='annotations/val2017')) -> None:
        self.config_path = config_path
        self.data_root = data_root
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.crop_size = crop_size
        self.scale = scale
        
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_train_path = os.path.join(file_dir,root_train_path)
        self.root_val_path = os.path.join(file_dir,root_val_path)
        self.img2subj_path = os.path.join(file_dir,img2subj_path)
        
        
        super().__init__(
            dataset_name='CocoStuff164k',
            tag = 'cocostuff',
            dataset_type='COCOStuffDataset'
        )
    
    def generate_meta(self,targets,out_dir=None,sample_num=500):
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
        summary['dataset'] = 'cocostuff'
        summary['tag'] = 'cocostuff'
        return summary