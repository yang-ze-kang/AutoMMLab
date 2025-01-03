_base_ = [
    '../_base_/models/fastfcn_r50-d32_jpu_psp.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_default.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
