import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_coco import DatasetBaseCoCo


class DatasetCityscapesDetection(DatasetBaseCoCo):

    CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    CLASSES_NAMES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    PALETTE = [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    
    def __init__(self,
                 data_root='sh1984:s3://openmmlab/datasets/segmentation/cityscapes',
                 raw_train_json_file='annos/cityscapes/instancesonly_filtered_gtFine_train.json',
                 raw_val_json_file='annos/cityscapes/instancesonly_filtered_gtFine_val.json') -> None:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        raw_train_json_file = os.path.join(file_dir,raw_train_json_file)
        raw_val_json_file = os.path.join(file_dir,raw_val_json_file)
        super().__init__(
            data_root=data_root,
            raw_train_json_file=raw_train_json_file,
            raw_val_json_file=raw_val_json_file,
            dataset_name='Cityscapes',
            tag='cityscapes',
            train_data_prefix=dict(img='leftImg8bit/train/'),
            val_data_prefix=dict(img='leftImg8bit/val/')
        )
    

if __name__=='__main__':
    objects = ["furniture"]
    dataset = DatasetCityscapesDetection()
    tar2label = dataset.search(objects)
    print(tar2label)