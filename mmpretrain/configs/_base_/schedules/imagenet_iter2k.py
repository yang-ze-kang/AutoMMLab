# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=5e-4))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=False, milestones=[500, 1200, 1800], gamma=0.1)

# train, val, test setting
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
