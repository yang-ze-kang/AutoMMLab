_base_ = [
    '../_base_/models/shufflenet_v1_1x.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_iter2k.py',
    '../_base_/default_runtime_iter2k.py'
]
