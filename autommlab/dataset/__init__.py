from .classification_imagenet import DatasetImageNetClassification
from .detection_coco import DatasetCocoDetection
from .detection_cityscapes import DatasetCityscapesDetection
from .detection_openimage import DatasetOpenimageDetection
from .detection_object365 import DatasetObject365Detection
from .segmentation_cityscapes import DatasetCityscapesSegmentation
from .pose_coco import DatasetPoseCoco
from .pose_ap10k import DatasetPoseAP10k


DATA_LIST = {
    'classification':['imagenet'],
    # 'detection':['object365','coco','cityscapes'],
    'detection':['coco'],
    'segmentation':['cityscapes'],
    "pose": ['coco','ap10k']
}

DATA_CLASS = {
    'classification':{
        'imagenet':DatasetImageNetClassification
    },
    'detection':{
        'object365':DatasetObject365Detection,
        'coco':DatasetCocoDetection,
        'cityscapes':DatasetCityscapesDetection,
        'openimage': DatasetOpenimageDetection
    },
    'segmentation':{
        'cityscapes':DatasetCityscapesSegmentation
    },
    'pose':{
        'coco':DatasetPoseCoco,
        'ap10k':DatasetPoseAP10k
    }
}
