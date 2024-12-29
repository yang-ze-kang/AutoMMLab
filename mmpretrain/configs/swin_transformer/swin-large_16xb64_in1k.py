_base_ = [
    '../_base_/models/swin_transformer/large_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_iter2k.py',
    '../_base_/default_runtime_iter2k.py'
]

optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0)
)
