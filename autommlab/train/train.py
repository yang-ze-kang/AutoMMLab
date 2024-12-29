# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from mmengine.config.config import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import numpy as np
import json
import torch

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # parser.add_argument('config', help='train config file path')
    parser.add_argument('config_dir', help='train config dir path')
    parser.add_argument('--pretrained_weights_dir',default='/data3/yangzekang/LLMaC/llama_inference/autommtrain_cls/pretrained_weights', help='train config dir path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(os.path.join(args.config_dir,'config.py'))
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.path.join(args.config_dir,'train_log')

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    with open(os.path.join(args.config_dir,'summary.json'),'r') as f:
        summary = json.load(f)
    if summary['model']['Weights'] is not None:
        weight_path = summary['model']['Weights']
        if args.pretrained_weights_dir is not None:
            weight_path = weight_path.replace('https://download.openmmlab.com',args.pretrained_weights_dir)
        cfg.load_from = weight_path

    # flush ann_file
    def flush_ann_file(cfg):
        for key in cfg:
            try:
                if isinstance(cfg[key], ConfigDict):
                    flush_ann_file(cfg[key])
                elif key in ['ann_file','bbox_file']:
                    cfg[key] = os.path.join(args.config_dir, 'meta', cfg[key])
            except:
                continue
        return cfg
    
    cfg = flush_ann_file(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    try:
        runner.train()
    except RuntimeError as e:
        with open(os.path.join(args.config_dir, 'train_results.json'),'w') as f:
            json.dump({"error":str(e)},f)
        return
    with open(os.path.join(args.config_dir, 'train_results.json'),'w') as f:
        path = os.path.join(args.config_dir,'train_log',runner.timestamp,'vis_data','scalars.json')
        ds = np.genfromtxt(path,dtype=str,delimiter='\n')
        metrics = []
        for d in ds:
            d = d.replace('NaN','0')
            d = eval(d)
            if 'lr' not in d:
                for metric in d:
                    if 'time' not in metric and 'step' not in metric:
                        metrics.append(metric)
                break
        results = []
        for d in ds:
            d = d.replace('NaN','0')
            d = eval(d)
            if 'lr' in d:
                continue
            res = {'step':d['step']}
            for metric in metrics:
                res.update({metric:d[metric]})
            results.append(res)
        json.dump(results,f,indent='\t')


if __name__ == '__main__':
    main()
