# Refers to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification

_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_small_imagenet.py',
    '../_base_/datasets/imagenet_bs128_mbv3.py',
    '../_base_/schedules/imagenet_iter2k.py',
    '../_base_/default_runtime_iter2k.py',
]
