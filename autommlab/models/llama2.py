import requests
import json

from autommlab.configs import URL_LLAMA


class LLaMA2():

    def __init__(self,url=URL_LLAMA):
        self.url = URL_LLAMA
        return
    
    def __call__(self,prompt,peft_model,temperature=0.6):
        data = {
            "prompt": prompt,
            "peft_model": peft_model,
            "temperature": temperature
        }
        response = requests.post(self.url, json=data)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {"error":"Call LLaMA error."}
