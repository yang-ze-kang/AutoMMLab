import json
import subprocess
import re

path = '/data3/yangzekang/LLMaC/llama_inference/models/detection_model_zoo.json'

with open(path,'r') as f:
    ds = f.read()
matches = re.findall(r'"Config":\s+"([^"]+)"', ds)

for config in matches:
    if 'yolox' in config:
        subprocess.run(['python',f'tools/analysis_tools/get_flops.py',f'{config}'])