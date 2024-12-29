import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nltk.corpus import wordnet
from base import DatasetBase
from pycocotools.coco import COCO
import json
import random

class DatasetBaseCoCo(DatasetBase):
    

    def __init__(self,
                data_root: str = '',
                raw_train_json_file: str =  '',
                raw_val_json_file: str =  '',
                input_size: tuple = (256, 256),
                dataset_name: str='',
                dataset_type: str='CocoDataset',
                train_data_prefix: dict=dict(img='train2017/'),
                val_data_prefix: dict=dict(img='val2017/'),
                tag: str='') -> None:
        self.data_root = data_root
        self.raw_train_json_file = raw_train_json_file
        self.raw_val_json_file = raw_val_json_file
        self.input_size = input_size
        self.heatmap_size = tuple(i//4 for i in input_size)
        super().__init__(
            dataset_name=dataset_name,
            tag=tag,
            dataset_type=dataset_type,
            train_data_prefix=train_data_prefix,
            val_data_prefix=val_data_prefix
        )

    def filter_target_objects(self, raw_json_file, out_json_file,targets=[], sample_num=None, coco_dict=None):
        if coco_dict==None or raw_json_file not in coco_dict:
            coco = COCO(raw_json_file)
            if coco_dict:
                coco_dict[raw_json_file] = coco
        else:
            print('cached')
            coco = coco_dict[raw_json_file]
        cat_ids = coco.getCatIds(catNms=targets)
        out_cats = coco.loadCats(cat_ids)
        
        img_ids = []
        for cat_id in cat_ids:
            ids = coco.getImgIds(catIds=cat_id)
            if sample_num is not None:
                ids = random.sample(ids,min(len(ids),sample_num))
            img_ids.extend(ids)
        ann_ids1 = coco.getAnnIds(catIds=cat_ids)
        img_ids = list(set(img_ids))
        out_imgs = coco.loadImgs(img_ids)

        ann_ids2 = coco.getAnnIds(imgIds=img_ids)
        ann_ids = list(set(ann_ids1) & set(ann_ids2))
        out_annos = coco.loadAnns(ann_ids)
        with open(raw_json_file,'r') as f, open(out_json_file,'w') as f_out:
            raw = json.load(f)
            if 'info' in raw:
                out = {
                    'info':raw['info'],
                    'licenses':raw['licenses'],
                    'images':out_imgs,
                    'annotations':out_annos,
                    'categories':out_cats
                }
            else:
                out = {
                    'images':out_imgs,
                    'annotations':out_annos,
                    'categories':out_cats
                }
            json.dump(out,f_out)

    def get_imgs_nums(self, file, targets=[]):
        coco = COCO(file)
        tar2num = {tar:0 for tar in targets}
        total_ids = set()
        for target in targets:
            ids = coco.getImgIds(catIds=coco.getCatIds(catNms=[target]))
            tar2num[target] = len(ids)
            total_ids = total_ids | set(ids)
        tar2num['total'] = len(total_ids)
        return tar2num
    
    def generate_meta(self, targets, out_dir=None, sample_num=200, coco=None, use_metainfo=True):
        if isinstance(targets, list):
            targets = self.search(targets)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        classes = []
        palette = []
        summary = {"target2labels":{}, "label2num":{}}
        targets_list = list(targets.keys())
        for obj in targets_list:
            summary['target2labels'][obj] = []
            for label in targets[obj]:
                summary['label2num'][label] = {'train':0,'val':0,'total':0}
                summary['target2labels'][obj].append(label)
                classes.append(label)
                if self.PALETTE is not None:
                    palette.append(self.PALETTE[self.CLASSES_NAMES.index(label)])
        if use_metainfo:
            if hasattr(self,'metainfo'):
                self.metainfo.update({
                    "classes":tuple(classes),
                    "palette":tuple(palette)
                })
            else:
                self.metainfo = {
                    "classes":tuple(classes),
                    "palette":tuple(palette)
                }
            summary['metainfo'] = self.metainfo
        if len(classes)<10:
            sample_num = 200
        else:
            sample_num = 20
        self.filter_target_objects(self.raw_train_json_file,os.path.join(out_dir,'train.json'),targets=classes,sample_num=sample_num*10,coco_dict=coco)
        self.filter_target_objects(self.raw_val_json_file,os.path.join(out_dir,'val.json'),targets=classes,sample_num=sample_num,coco_dict=coco)
        self.train_path = 'train.json'
        self.val_path = 'val.json'
        tar2num_train = self.get_imgs_nums(os.path.join(out_dir,'train.json'),classes)
        tar2num_val = self.get_imgs_nums(os.path.join(out_dir,'val.json'),classes)
        for cls in classes:
            summary['label2num'][cls]['train'] = tar2num_train[cls]
            summary['label2num'][cls]['val'] = tar2num_val[cls]
            summary['label2num'][cls]['total'] = tar2num_train[cls]+tar2num_val[cls]
        summary['label2num']['total'] = {}
        summary['label2num']['total']['train'] = tar2num_train['total']
        summary['label2num']['total']['val'] = tar2num_val['total']
        summary['label2num']['total']['total'] = tar2num_train['total']+tar2num_val['total']
        summary['num_classes'] = len(classes)
        summary['dataset'] = self.dataset_name
        summary['tag'] = self.tag
        return summary