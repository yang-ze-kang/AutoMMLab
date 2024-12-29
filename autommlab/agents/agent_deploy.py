import subprocess
import os

class AgentDeploy():

    def __init__(self) -> None:
        self.onnx_configs={'classification':'mmdeploy/configs/mmpretrain/classification_onnxruntime_dynamic.py',
                           'detection':'mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py',
                           'segmentation':'mmdeploy/configs/mmseg/segmentation_onnxruntime_dynamic.py',
                           'pose':'mmdeploy/configs/mmpose/pose-detection_onnxruntime_static.py'}

    def run(self, cfg_path, model_path, out_dir, task='classification',mode='onnx'):
        path = os.path.join(out_dir,'deploy')
        if mode=='onnx':
            cmd_command = f"python mmdeploy/tools/deploy.py \
                {self.onnx_configs[task]} \
                {cfg_path} \
                {model_path} \
                autommlab/examples/cats/ILSVRC2012_val_00000130.JPEG \
                --work-dir {path} \
                --show \
                --dump-info \
                --device cuda:0"
            completed_process = subprocess.run(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print(completed_process)
            if completed_process.returncode == 0:
                return path
            else:
                return None