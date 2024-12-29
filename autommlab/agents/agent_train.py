import argparse
import os
import os.path as osp
from copy import deepcopy
import numpy as np
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
import json

from autommlab.utils.util import array_to_markdown_table


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False
        if args.no_persistent_workers:
            cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


class AgentTrainModel():

    task2metric = {
        'classification':'multi-label/accuracy_top1',
        'detection':'coco/bbox_mAP',
        'segmentation':'mIoU',
        'keypoint':'coco/AP'
    }
    
    def __init__(self) -> None:
        self.training = False

    def run(self,cfg_path,log_dir):
        args = parse_args()
        cfg = Config.fromfile(cfg_path)
        cfg = merge_args(cfg, args)
        cfg.work_dir = os.path.join(log_dir,'train_logs')
        os.makedirs(cfg.work_dir,exist_ok=True)
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)
        self.training = True
        runner.train()
        self.training = False
        return True

    
    def get_best_result(self, log_dir):
        path = os.path.join(log_dir,'train_results.json')
        with open(path,'r') as f:
            ds = json.load(f)
        if 'error' in ds:
            return ds
        metrics = []
        for metric in ds[0]:
            if 'time' not in metric and 'step' not in metric:
                metrics.append(metric)
        self.metrics = metrics
        results = []
        for d in ds:
            res = []
            res.append(d['step'])
            for metric in metrics:
                res.append(d[metric])
            results.append(res)
        results = np.array(results)
        ind = np.lexsort(keys=[results[:,i] for i in range(len(results[0]))][::-1])[::-1]
        best = results[ind[0]]
        epoch = int(best[0])
        res = {'step':epoch,'model_path':os.path.join(log_dir,'train_log',f"iter_{epoch}.pth")}
        for i, metric in enumerate(metrics):
            res[metric] = best[i+1]
        return res
    
    def format_metric_result(self,res):
        s = "Model training is completed, the best epoch is:\n"
        rows = []
        row = ["iter"]
        for metric in self.metrics:
            row.append(metric)
        rows.append(row)
        row = [res['step']]
        for metric in self.metrics:
            row.append(res[metric])
        rows.append(row)
        return s+array_to_markdown_table(rows)


if __name__=='__main__':
    agent = AgentTrainModel()
    res = agent.get_best_result([{"name":'accuracy'},{"name":'f1-score'}],'/data3/yangzekang/LLMaC/llama_inference/logs/2023-09-06-14:11')
    print(res)



