from mmengine.config import Config
from utils.util import flush_config, array_to_markdown_table
from thefuzz import process
import numpy as np
import os

from autommlab.dataset import DATA_LIST,DATA_CLASS

class AgentDataset():

    def __init__(self) -> None:
        self.mode = 'none'
        pass

    def run_single(self,parse,out_dir,mode='classification',coco=None):
        self.mode = mode
        errors = {}
        if len(parse['specific']) != 0:
            for spc in parse['specific']:
                cls = process.extract(spc,DATA_LIST[mode],limit=5)[0][0]
                self.agent = DATA_CLASS[mode][cls]()
                tar2labels = self.agent.search(parse['object'])
                if 'error' not in tar2labels:
                    if mode=='detection':
                        res = self.agent.generate_meta(tar2labels,os.path.join(out_dir, 'meta'),coco=coco)
                    else:
                        res = self.agent.generate_meta(tar2labels,os.path.join(out_dir, 'meta'))
                    if 'error' not in res:
                        return res
                    else:
                        errors[cls] = res
                else:
                    errors[cls] = tar2labels['error']
        for cls in DATA_LIST[mode]:
            self.agent = DATA_CLASS[mode][cls]()
            tar2labels = self.agent.search(parse['object'])
            if 'error' not in tar2labels:
                if mode=='detection':
                    res = self.agent.generate_meta(tar2labels,os.path.join(out_dir, 'meta'),coco=coco)
                else:
                    res = self.agent.generate_meta(tar2labels,os.path.join(out_dir, 'meta'))
                if 'error' not in res:
                    return res
                else:
                    errors[cls] = res
            else:
                errors[cls] = tar2labels['error']
        return {'error':errors}
    
    
    def generate_config(self,summary,cfg=None, out_dir=None):
        if cfg is None:
            cfg = Config.fromfile(self.agent.config_path)

        cfg = flush_config(cfg,'data_root',self.agent.data_root)
        cfg = flush_config(cfg,'num_classes',summary['num_classes'])

        if hasattr(cfg['train_dataloader']['dataset'],'dataset'):
            cfg['train_dataloader']['dataset']['dataset']['metainfo'] = self.agent.metainfo
            cfg['train_dataloader']['dataset']['dataset']['ann_file'] = self.agent.train_path
        else:
            cfg['train_dataloader']['dataset']['metainfo'] = self.agent.metainfo
            cfg['train_dataloader']['dataset']['ann_file'] = self.agent.train_path
        cfg['val_dataloader']['dataset']['metainfo'] = self.agent.metainfo
        cfg['test_dataloader']['dataset']['metainfo'] = self.agent.metainfo
        cfg['val_dataloader']['dataset']['ann_file'] = self.agent.val_path
        cfg['test_dataloader']['dataset']['ann_file'] = self.agent.val_path

        if self.mode in ['detection', 'pose']:
            cfg['val_evaluator']['ann_file'] = self.agent.val_path
            cfg['test_evaluator']['ann_file'] = self.agent.val_path
            
        if self.mode in ['detection']:
            cfg['train_dataloader']['dataset']['type'] = self.agent.dataset_type
            cfg['val_dataloader']['dataset']['type'] = self.agent.dataset_type
            cfg['test_dataloader']['dataset']['type'] = self.agent.dataset_type
            cfg['train_dataloader']['dataset']['data_prefix'] = self.agent.train_data_prefix
            cfg['val_dataloader']['dataset']['data_prefix'] = self.agent.val_data_prefix
            cfg['test_dataloader']['dataset']['data_prefix'] = self.agent.val_data_prefix
            if self.agent.dataset_type=="OpenImagesDataset":
                cfg['train_dataloader']['dataset']['label_file'] = self.agent.label_file
                cfg['val_dataloader']['dataset']['label_file'] = self.agent.label_file
                cfg['test_dataloader']['dataset']['label_file'] = self.agent.label_file
                cfg['train_dataloader']['dataset']['hierarchy_file'] = self.agent.hierarchy_file
                cfg['val_dataloader']['dataset']['hierarchy_file'] = self.agent.hierarchy_file
                cfg['test_dataloader']['dataset']['hierarchy_file'] = self.agent.hierarchy_file
                cfg['train_dataloader']['dataset']['meta_file'] = self.agent.meta_file
                cfg['val_dataloader']['dataset']['meta_file'] = self.agent.train_meta_file
                cfg['test_dataloader']['dataset']['meta_file'] = self.agent.val_meta_file

        # if self.mode in ['segmentation']:
        #     flush_config(cfg,'crop_size',self.agent.crop_size)


        if out_dir is not None:
            cfg.dump(os.path.join(out_dir,'config_dataset.py'))
        return cfg

    
    def format_result(self,result):
        if result['tag'] in ['in1k']:
            res = "I found pictures containing "
            targets = list(result['target2labels'].keys())
            res += targets[0]
            for i in range(1,len(targets)-1):
                res += f", {targets[i]}"
            res += f" and {targets[-1]}"
            res+=f" from the public dataset {result['dataset']}, and built the training set and a test set:\n"
            
            rows = []
            row = [""]
            for tar,labels in result['target2labels'].items():
                item = f"{tar}({labels[0]}"
                for label in labels[1:]:
                    item+=f',{label}'
                item+=")"
                row.append(item)
            row.append("sum number")
            rows.append(row)
            matrix = np.zeros((3,len(result['target2num'])+1),dtype=int)
            for i, (tar, val) in enumerate(result['target2num'].items()):
                matrix[0,i] = int(val['train'])
                matrix[1,i] = int(val['val'])
                matrix[2,i] = int(val['total'])
                matrix[0,len(result['target2num'])] += int(val['train'])
                matrix[1,len(result['target2num'])] += int(val['val'])
                matrix[2,len(result['target2num'])] += int(val['total'])
            row = ['train number']
            row.extend(matrix[0])
            rows.append(row)
            row = ['test number']
            row.extend(matrix[1])
            rows.append(row)
            row = ['sum number']
            row.extend(matrix[2])
            rows.append(row)

            res+=array_to_markdown_table(rows)
        elif result['tag'] in ['coco','lvis','cityscapes','ap10k','object365']:
            res = "I found pictures containing "
            targets = list(result['label2num'].keys())
            res += targets[0]
            for i in range(1,len(targets)-1):
                res += f", {targets[i]}"
            res += f" and {targets[-1]}"
            res+=f" from the public dataset {result['dataset']}, and built the training set and a test set:\n"
            
            rows = []
            row = [""]
            row.extend(result['label2num'].keys())
            rows.append(row)
            matrix = np.zeros((3,len(result['label2num'])),dtype=int)
            for i, (tar, val) in enumerate(result['label2num'].items()):
                matrix[0,i] = int(val['train'])
                matrix[1,i] = int(val['val'])
                matrix[2,i] = int(val['total'])
            row = ['train number']
            row.extend(matrix[0])
            rows.append(row)
            row = ['test number']
            row.extend(matrix[1])
            rows.append(row)
            row = ['sum number']
            row.extend(matrix[2])
            rows.append(row)

            res+=array_to_markdown_table(rows)
        return res
        


                