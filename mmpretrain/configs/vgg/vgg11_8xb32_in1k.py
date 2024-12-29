_base_ = [
    '../_base_/models/vgg11.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_iter2k.py',
    '../_base_/default_runtime_iter2k.py',
]
