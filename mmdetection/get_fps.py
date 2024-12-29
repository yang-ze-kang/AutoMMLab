import json
import subprocess
import re

path = '/data3/yangzekang/LLMaC/llama_inference/models/detection_model_zoo.json'

with open(path,'r') as f:
    ds = f.read()
matches = re.findall(r'"Config":\s+"([^"]+)"', ds)

for config in matches:
    if 'yolo' in config and 'd53' not in config:
        subprocess.run(['python',f'-m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=29501 tools/analysis_tools/benchmark.py',f'{config}'])