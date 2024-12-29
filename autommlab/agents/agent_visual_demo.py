import json
import numpy as np
import os
import torch
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmdeploy_runtime import Classifier
import cv2
import json

from autommlab.configs import DATASET_ZOO

class DemoVisualAgent():

    dataset_root={
        'ImageNet1k':DATASET_ZOO['ImageNet'],
        'CoCo':DATASET_ZOO['COCO']
    }
    onnx_configs={
        'classification':'mmdeploy/configs/mmpretrain/classification_onnxruntime_dynamic.py',
        'detection':'mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py',
        'segmentation':'mmdeploy/configs/mmseg/segmentation_onnxruntime_dynamic.py',
        'pose':'mmdeploy/configs/mmpose/pose-detection_onnxruntime_static.py'
    }
    def __init__(self):
        pass
        

    def predict(self,log_dir,task,img_path,out_path):
        backend_model = [os.path.join(log_dir,'deploy','end2end.onnx')]
        deploy_cfg = self.onnx_configs[task]
        model_cfg = os.path.join(log_dir,'config.py')
        device = 'cpu'
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        model = task_processor.build_backend_model(backend_model)
        input_shape = get_input_shape(deploy_cfg)
        model_inputs, _ = task_processor.create_input(img_path, input_shape)
        with torch.no_grad():
            result = model.test_step(model_inputs)
        task_processor.visualize(
            image=img_path,
            model=model,
            result=result[0],
            window_name='visualize',
            output_file=out_path)
    
    def start(self,log_dir,task):
        try:
            with open(os.path.join(log_dir,'summary.json'),'r') as f:
                ds = json.load(f)
            dataset = ds['data']['dataset']
            out_dir = os.path.join(log_dir,'predictions')
            os.makedirs(out_dir,exist_ok=True)
            out_paths = []
            in_paths = []
            if task.lower()=='classification':
                img_paths = np.genfromtxt(os.path.join(log_dir,'meta','val.txt'),dtype=str,delimiter=' ')[:,0]
                img_paths = np.random.choice(img_paths,5,replace=False)
            elif task.lower()=='detection':
                with open(os.path.join(log_dir,'meta','val.json'),'r') as f:
                    ds = json.load(f)
                    ds = np.random.choice(ds['images'],5,replace=False)
                    img_paths = [item['file_name'] for item in ds]
            for img_path in img_paths:
                out_path = os.path.join(out_dir,img_path)
                self.predict(log_dir,task,self.dataset_root[dataset]+img_path,out_path)
                out_paths.append(out_path)
        except:
            print('inference error')
            return []

        return out_paths

