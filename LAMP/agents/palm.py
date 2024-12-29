import re
import google.generativeai as genai
import json
import os
import requests
from prompts import init_prompt, example_prefix_prompt, examples_prompt, req_prefix_prompt
from dotenv import load_dotenv

PROXY = "10.155.176.131:8888"

def flush_proxy():
    if 'HTTP_PROXY' in os.environ:
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        del os.environ['HTTPS_PROXY']

def set_proxy():
    flush_proxy()
    os.environ["HTTP_PROXY"] = PROXY
    os.environ["HTTPS_PROXY"] = PROXY

def get_ip_location(proxies={"http": "127.0.0.1:7890", "https": "127.0.0.1:7890"}):
    try:
        response = requests.get("https://ipinfo.io")
        data = response.json()
        
        ip = data.get("ip")
        city = data.get("city")
        region = data.get("region")
        country = data.get("country")
        location = data.get("loc")
        
        print(f"IP: {ip}")
        print(f"Location: {location}")
        print(f"City: {city}")
        print(f"Region: {region}")
        print(f"Country: {country}")
        return country
        
    except Exception as e:
        print("An error occurred:", e)
        return False

class RUPaLM2():

    def __init__(self,model_name='models/text-bison-001', api_key="", log_dir='logs',iter=1,example_num=0) -> None:
        genai.configure(api_key=api_key)
        assert example_num<=len(examples_prompt)
        self.iter = iter
        self.model_name = model_name
        self.example_num = example_num
        self.log_dir = os.path.join(log_dir,f"en{example_num}")
        os.makedirs(self.log_dir,exist_ok=True)

    
    def run(self, query, temperature=0.1):
        prompt = init_prompt
        if self.example_num>0:
            prompt=prompt+'\n'+example_prefix_prompt+'\n'
            for i in range(self.example_num):
                prompt = prompt+examples_prompt[i]+'\n'
        prompt = prompt+'\n'+req_prefix_prompt+query
        response = genai.generate_text(model=self.model_name,prompt=prompt,temperature=temperature)
        print(response)
        response = response.result
        if self.log_dir is not None:
            with open(os.path.join(self.log_dir,f"parse_{self.iter}_t{temperature}.txt"),'w') as f:
                f.write(str(response)+'\n')
            self.iter+=1
        return self.parse_result(response)
    
    def parse_result(self,res):
        s = res
        for m in re.findall("###.*###", res):
            s = s.replace(m,"")
        s = s.strip()
        data = json.loads(s)
        return data


if __name__=='__main__':
    set_proxy()
    get_ip_location()
    load_dotenv()
    model = 'palm2-bison'
    path = '/home/SENSETIME/yangzekang/LLMaC/LLMaC/generate_parse_test_data/data_v2/reqparse_test.json'
    agent = RUPaLM2()
    query = 'I am a zookeeper and I need a model that can classify different types of big cats, including: lions, tigers, leopards, and cheetahs. The model should be based on EfficientNet and the number of parameters should not exceed 25M. The model should be able to run on a CPU and the accuracy should be greater than 95%. This model should be deployed using ONNX Runtime as the inference engine.'
    res = agent.run(query)
    print(res)