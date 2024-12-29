import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base import DatasetBase
from nltk.corpus import wordnet 
import csv
import numpy as np
import random
import shutil

from autommlab.configs import DATASET_ZOO

class DatasetOpenimageDetection(DatasetBase):

    CLASSES = None

    CLASSES_NAMES = None
    
    PALETTE = None

    def __init__(self,
                data_root=DATASET_ZOO['openimage'],
                train_raw_ann_file='annos/openimage/oidv6-train-annotations-bbox.csv',
                val_raw_ann_file='annos/openimage/validation-annotations-bbox.csv',
                raw_label_file='annos/openimage/class-descriptions-boxable.csv',
                train_label2num_file = 'annos/openimage/label2num_train.txt',
                val_label2num_file = 'annos/openimage/label2num_val.txt',
                raw_hierarchy_file='annos/openimage/bbox_labels_600_hierarchy.json',
                raw_meta_file='annos/openimage/train-image-metas.pkl',
                train_image_level_ann_file=None,
                val_image_level_ann_file='annotations/validation-annotations-human-imagelabels-boxable.csv') -> None:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_raw_ann_file = os.path.join(file_dir,train_raw_ann_file)
        self.val_raw_ann_file = os.path.join(file_dir,val_raw_ann_file)
        self.raw_label_file = os.path.join(file_dir,raw_label_file)
        self.raw_hierarchy_file = os.path.join(file_dir,raw_hierarchy_file)
        self.raw_meta_file = os.path.join(file_dir,raw_meta_file)
        self.train_label2num_file = os.path.join(file_dir,train_label2num_file)
        self.val_label2num_file = os.path.join(file_dir,val_label2num_file)
        
        self.data_root = data_root

        classes_names, label_id_mapping = self._parse_label_file(self.raw_label_file)
        self.CLASSES = classes_names
        self.CLASSES_NAMES = classes_names
        self.label_id_mapping = label_id_mapping
    
        super().__init__(
            dataset_name='openimage',
            tag='openimage',
            dataset_type='OpenImagesDataset',
            train_data_prefix=dict(img='OpenImages/train/'),
            val_data_prefix=dict(img='OpenImages/validation/')
        )

    def _parse_label_file(self, label_file: str) -> tuple:
        """Get classes name and index mapping from cls-label-description file.

        Args:
            label_file (str): File path of the label description file that
                maps the classes names in MID format to their short
                descriptions.

        Returns:
            tuple: Class name of OpenImages.
        """

        index_list = []
        classes_names = []
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                # self.cat2label[line[0]] = line[1]
                classes_names.append(line[1])
                index_list.append(line[0])
        index_mapping = {index: i for i, index in enumerate(index_list)}
        return classes_names, index_mapping
        

    def filter_target_objects(self, raw_ann_file, out_ann_file,targets=[], sample_num=None):
        outs = []
        tar2imgs = {tar:set() for tar in targets}
        ds = np.genfromtxt(raw_ann_file,delimiter=',',dtype=str,skip_header=1)
        random.shuffle(ds)
        for d in ds:
            label = self.CLASSES_NAMES[self.label_id_mapping[d[2]]]
            if label in targets:
                if len(tar2imgs[label])>=sample_num:
                    continue
                outs.append(','.join(d)+'\n')
                tar2imgs[label].add(d[0])
        ids = list(range(len(outs)))
        with open(out_ann_file,'w') as f:
            f.write("ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n")
            for id in ids:
                f.write(outs[id])
        tar2num = {tar:0 for tar in targets}
        for key in tar2imgs:
            tar2num[key] = len(tar2imgs[key])
        tar2num['total'] = len(ids)
        print(tar2num)
        return tar2num



    def get_imgs_nums(self, file, targets=[]):
        tar2num = {tar:0 for tar in targets}
        ds = np.genfromtxt(file, delimiter=',',dtype=str,skip_header=1)
        for d in ds:
            if d[0] in targets:
                tar2num[d[0]] = d[1]
        return tar2num
    
    def summary_targets_nums(self, targets=[]):
        classes = []
        for obj in list(targets.keys()):
            for label in targets[obj]:
                classes.append(label)
        tar2num_train = self.get_imgs_nums(self.train_label2num_file,classes)
        tar2num_val = self.get_imgs_nums(self.val_label2num_file,classes)
        return {'train':tar2num_train, 'val':tar2num_val}
    
    def generate_meta(self, targets, out_dir=None, sample_num=20):
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
        tar2num_train = self.get_imgs_nums(self.train_label2num_file,classes)
        tar2num_val = self.get_imgs_nums(self.val_label2num_file,classes)
        if set(tar2num_train.keys())!=set(tar2num_val.keys()):
            classes = list(set(tar2num_train.keys())&set(tar2num_val.keys()))
            if len(classes)==0:
                return {'error':'not found'}
        tar2num_train = self.filter_target_objects(self.train_raw_ann_file,os.path.join(out_dir,'train.csv'),targets=classes,sample_num=sample_num*10)
        tar2num_val  = self.filter_target_objects(self.val_raw_ann_file,os.path.join(out_dir,'val.csv'),targets=classes,sample_num=sample_num)
        shutil.copyfile(self.raw_label_file,os.path.join(out_dir,'class-descriptions-boxable.csv'))
        shutil.copyfile(self.raw_hierarchy_file,os.path.join(out_dir,'bbox_labels_600_hierarchy.json'))
        shutil.copyfile(self.raw_meta_file,os.path.join(out_dir,'train-image-metas.pkl'))
        self.train_path = 'train.csv'
        self.val_path = 'val.csv'
        self.label_file = 'class-descriptions-boxable.csv'
        self.hierarchy_file = 'bbox_labels_600_hierarchy.json'
        self.meta_file = 'train-image-metas.pkl'
        # tar2num_train = self.get_imgs_nums(os.path.join(out_dir,'train.json'),classes)
        # tar2num_val = self.get_imgs_nums(os.path.join(out_dir,'val.json'),classes)
        if set(tar2num_train.keys())!=set(tar2num_val.keys()):
            return {'error':'not found'}
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