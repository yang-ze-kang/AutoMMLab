_base_ = './fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py'
model = dict(backbone=dict(depth=101))
