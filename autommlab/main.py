import multiprocessing
from multiprocessing import Process
import socket
import subprocess
import datetime
import openai
import json
import gradio as gr
import os
import argparse
import shutil
import copy
import time
import numpy as np

from mmengine.config import Config

from autommlab.agents.agent_ru import AgentRU
from autommlab.agents.agent_dataset import AgentDataset
from autommlab.agents.agent_model import AgentModel
from autommlab.agents.agent_hpo import AgentHPO
from autommlab.agents.agent_train import AgentTrainModel
from autommlab.agents.agent_deploy import AgentDeploy
from autommlab.agents.agent_visual_demo import DemoVisualAgent
from autommlab.configs import *
from autommlab.utils.util import flush_config, array_to_markdown_table


class AutoMMLab():

    def __init__(self,
        ru_model = RU_MODEL,
        hpo_model = HPO_MODEL,
        hpo_max_try = HPO_MAX_TRY
    ) -> None:
        self.agent_ru = AgentRU(ru_model)
        self.agent_dataset = AgentDataset()
        self.agent_model = AgentModel()
        self.agent_hpo = AgentHPO()
        self.agent_train = AgentTrainModel()
        self.agent_deploy = AgentDeploy()
        self.agent_visual_demo = DemoVisualAgent()
        self.hpo_max_try = hpo_max_try
    
    def save_summary(self, summary,out_dir):
        with open(os.path.join(out_dir,'summary.json'),'w') as f:
            json.dump(summary, f, indent='\t')

    def run(self, input='',parse='', log_dir=None):
        if log_dir is None:
            current_time = datetime.datetime.now()
            id = current_time.strftime("%Y%m%d%H%M")
            log_dir = os.path.join(os.getcwd(), 'logs', id)
        bag_dir = os.path.join(log_dir,'bag')
        os.makedirs(bag_dir,exist_ok=True)

        summary = {}

        ###############
        # ===parse=== #
        ###############
        parse = self.agent_ru(input)
        if 'error' in parse:
            yield parse['error']
            return
        s = json.dumps(parse,indent='\t')
        yield f"""Requirement parsing result:\n```JSON\n{s}\n```"""
        task = parse['model']['task']

        ms = {metric['name']:metric['value'] for metric in parse['model']['metrics']}
        if self.agent_ru.task2metric[task] in list(ms.keys()):
            target_acc = ms[self.agent_ru.task2metric[task]]
        else:
            target_acc = 0
        
        ##################
        # ====dataset=== #
        ##################
        self.summary_data = self.agent_dataset.run_single(parse['data'], bag_dir, parse['model']['task'])
        summary['data'] = self.summary_data
        if 'error' in self.summary_data:
            self.save_summary(summary, log_dir)
            tars = []
            for key in self.summary_data['error']:
                tars = self.summary_data['error'][key]
                break
            s = ', '.join(tars)
            yield f"The following images are not found in the database: {s}"
            return
        parse['num_classes'] = self.summary_data['num_classes']
        print(self.summary_data)
        yield self.agent_dataset.format_result(self.summary_data)

        ###############
        # ===model=== #
        ###############
        summary_model, format_print = self.agent_model.run_single(parse['model'], bag_dir, data_tag=self.summary_data['tag'])
        summary['model'] = summary_model
        if 'error' in summary_model:
            self.save_summary(summary, log_dir)
            yield summary_model['error']
            return
        yield format_print
        
        config_path = self.merge_config(parse, bag_dir)
        self.save_summary(summary, log_dir)

        #########################
        #  ===HPO and Train===  #
        #########################
        command = f"tensorboard --logdir={log_dir} --port={TENSORBOARD_PORT} --host=0.0.0.0"
        tensorboard = subprocess.Popen(command,shell=True)
        hpo_try, acc, hpo_res = 0, 0, []
        convs = self.init_hpo_convs(task, summary)
        while hpo_try==0 or (hpo_try<self.hpo_max_try and acc<target_acc):
            hpo_try+=1
            # hpo
            hp = self.agent_hpo(convs, task)
            convs[-1]['assistant'] = json.dumps(hp)
            hp = self.get_hp(hp)
            s = json.dumps(hp,indent='\t')
            if hpo_try==1:
                yield f'Model is training !\nOpen <http://{IP_ADDRESS}:{TENSORBOARD_PORT}/#timeseries> to see the training process!' 
            yield f"""Hyperparameter optimization round {hpo_try}:\n```JSON\n{s}\n```"""
            bag_hp_dir = bag_dir+f'_hp{hpo_try}'
            os.makedirs(bag_hp_dir,exist_ok=True)
            shutil.copytree(bag_dir,bag_hp_dir,dirs_exist_ok=True)
            shutil.copy(os.path.join(log_dir,'summary.json'),os.path.join(bag_hp_dir,'summary.json'))
            with open(os.path.join(bag_hp_dir,'hyper_params.json'),'w') as f:
                json.dump(hp,f,indent='\t')
            config = Config.fromfile(os.path.join(bag_hp_dir,'config.py'))
            config_new = copy.deepcopy(config)
            config_new['default_hooks']['checkpoint']['interval'] = 2000
            config_new['default_hooks']['checkpoint']['save_last'] = True
            config_new['train_cfg']['max_iters'] = hp['iters']
            config_new['train_cfg']['val_interval'] = hp['iters']
            config_new['optim_wrapper']['optimizer'] = hp['optimizer']
            if task in ['detection'] and hp['optimizer']['lr']>=0.001:
                hp['lrs'] = [dict(type='LinearLR',start_factor=0.01, by_epoch=False, begin=0, end=500), hp['lrs']]
            elif task in ['segmentation'] and hp['optimizer']['lr']>=0.001:
                hp['lrs'] = [dict(type='LinearLR',start_factor=0.01, by_epoch=False, begin=0, end=100), hp['lrs']]
            config_new['param_scheduler'] = hp['lrs']
            config_new['train_dataloader']['batch_size'] = hp['bs']
            config_new.dump(os.path.join(bag_hp_dir,'config.py'))
            with open(os.path.join(log_dir,f'hpo_{hpo_try}a.json'),'w') as f:
                json.dump(convs,f,indent='\t')
            
            ## Train
            print('Start train!')
            bash_command = f"autommlab/train/dist_train.sh {bag_hp_dir} {TRAIN_GPU_NUM} --work-dir {os.path.join(bag_hp_dir,'train_log')}"
            process = subprocess.Popen(bash_command, shell=True, executable="/bin/bash")
            print(f'Open {IP_ADDRESS}:{TENSORBOARD_PORT}/#timeseries to see train process!')
            
            process.wait()
            if process.returncode != 0:
                print('An error occurred during training.')
                yield "An error occurred during training."
                return

            train_res = self.agent_train.get_best_result(bag_hp_dir)
            if 'error' in train_res:
                yield train_res['error']
            else:
                yield self.agent_train.format_metric_result(train_res) 
                acc = train_res[self.agent_train.task2metric[task]]
                hpo_res.append([acc,train_res['model_path']])
                if acc<target_acc:
                    convs.append({"user": f"The model trained with this set of hyperparameters has an {self.agent_ru.task2metric[task]} of {acc} on the test dataset. Please provide a better set of hyperparameters.",})
                    with open(os.path.join(log_dir,f'hpo_{hpo_try}b.json'),'w') as f:
                        json.dump(convs,f,indent='\t')
        hpo_res = np.array(hpo_res)
        id,hpo_res_s = self.format_hpo_result(hpo_res,self.agent_ru.task2metric[task])
        yield hpo_res_s
        ################
        # ===deploy=== #
        ################
        deploy_dir = self.agent_deploy.run(config_path,hpo_res[id,1], log_dir, task=parse['model']['task'])
        if deploy_dir == None:
            yield 'Deploy failed!'
            return 
        else:
            yield "Deploy finished!"

        # Visual
        if parse['model']['task'] in ['classification','detection']:
            predict_paths = self.agent_visual_demo.start(log_dir,parse['model']['task'])
            yield predict_paths
        self.save_summary(summary, log_dir)     

    
    def merge_config(self, cfg, log_dir):
        path = os.path.join(log_dir,'config.py')
        model = Config.fromfile(os.path.join(log_dir,'config_raw.py'))
        model = self.agent_dataset.generate_config(self.summary_data, cfg=model)
        model = flush_config(model,'num_classes',cfg['num_classes'])
        if cfg['model']['task'] == 'classification':
            model['val_evaluator']['type'] = 'MultiLabelMetric'
            model['val_evaluator']['topk'] = 1
            model['val_evaluator']['items'] = ['precision','recall','f1-score','accuracy']
            model['test_evaluator'] = model['val_evaluator']

        # demo
        model['default_hooks']['logger']['interval'] = 10 
        model['train_cfg']['max_iters'] = 200
        model['train_cfg']['val_interval'] = 40
        model['default_hooks']['checkpoint'] = dict(by_epoch=False, interval=2000, type='CheckpointHook')
        model['visualizer'] ['vis_backends'] = [
                dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
            ]
        model['train_dataloader']['batch_size'] = 4
        model['val_dataloader']['batch_size'] = 4
        model['test_dataloader']['batch_size'] = 4

        model.dump(path)
        return path

    def init_hpo_convs(self,task,summary):
        task2acc = {"classification":"accuracy","detection":"mAP","segmentation":"mIoU","keypoint":"mAP"}
        task2acc2 = {"classification":"Accuracy","detection":"box AP","segmentation":"mIoU","keypoint":"AP"}
        convs = [{"user":{}}]
        if task in ['detection','segmentation','keypoint']:
            convs[0]['user']['data']={"train_num":summary['data']['target2num']['total']['train'],"val_num":summary['data']['target2num']['total']['val'],"num_classes":summary['data']['num_classes'],"dataset":summary['data']['dataset']}
        else:
            convs[0]['user']['data']={"num_classes":summary['data']['num_classes'],"dataset":summary['data']['dataset']}
        convs[0]["user"]['model']={"name":summary['model']['Model'],"params(M)":summary['model']['Params(M)'],"flops(G)":summary['model']['Flops(G)'],f"{task2acc[task]}":summary['model'][task2acc2[task]]}
        return convs


    def get_hp(self,hp):
        if hp=='error' or 'optimizer' not in hp or 'lr schedule' not in hp or 'iters' not in hp or 'batch size' not in hp:
            return None
        if hp['optimizer'] in ['SGD','RMSprop']:
            optimizer = dict(
                type=hp['optimizer'],
                lr=hp['learning rate'],
                momentum=0.9,
                weight_decay=hp['weight decay']
            )
        elif hp['optimizer'] in ['Adam','AdamW']:
            optimizer = dict(
                type=hp['optimizer'],
                lr=hp['learning rate'],
                weight_decay=hp['weight decay']
            )
        if hp['lr schedule'] == 'MultiStepLR':
            lrs = dict(
                type=hp['lr schedule'],
                milestones=[int(hp['iters']*0.5),int(hp['iters']*0.75)],
                gamma=0.1,
                by_epoch=False
            )
        elif hp['lr schedule'] == 'CosineAnnealingLR':
            lrs = dict(
                type=hp['lr schedule'],
                T_max=hp['iters'],
                by_epoch=False
            )
        elif hp['lr schedule'] == 'StepLR':
            lrs = dict(
                type=hp['lr schedule'],
                step_size=50,
                gamma=0.96,
                by_epoch=False
            )
        elif hp['lr schedule'] == 'PolyLR':
            lrs = dict(
                type=hp['lr schedule'],
                power=0.9,
                by_epoch=False
            )
        hp_n = {
            'iters':hp['iters'],
            'bs':hp['batch size'],
            'optimizer':optimizer,
            'lrs':lrs
        }
        return hp_n
    
    def format_hpo_result(self,res,metric):
        res = np.array(res)
        id = np.argsort(res[:,0])[-1]
        s = f"Model hyperparameter optimization is completed, the best {metric} is: {round(float(res[id,0]),2)}\n"
        rows = [["round",f"{metric}"]]
        for i,acc in enumerate(res[:,0]):
            rows.append([str(i+1),f"{round(float(acc),2)}"])
        return int(id), s+array_to_markdown_table(rows,int(id)+1)

if __name__=='__main__':
    multiprocessing.set_start_method("spawn")
    llmac = AutoMMLab()
    with gr.Blocks(title='AutoMMLab',enable_queue=True) as demo:
        with gr.Row():
            chatbot = gr.Chatbot().style(height=650)
        with gr.Row():
            with gr.Column(scale=0.8):
                msg = gr.Textbox()
            with gr.Column(scale=0.2):
                clear = gr.Button("Clear")

        async def respond(message, chat_history):
            chat_history.append((message,None))
            yield gr.update(value="",interactive=False), chat_history
            for bot_message in llmac.run(message):
                if isinstance(bot_message,list):
                    for message in bot_message:
                        s = [message]
                        chat_history.append((None, s))
                        yield "", chat_history
                else:
                    chat_history.append((None, bot_message))
                    yield "", chat_history
            yield gr.update(interactive=True), chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot],queue=True)
        clear.click(lambda: None, None, chatbot, queue=False)
        demo.queue()
        demo.launch(server_port=10065, server_name="0.0.0.0")