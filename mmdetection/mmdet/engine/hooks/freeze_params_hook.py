from mmengine.hooks import Hook

from mmdet.registry import HOOKS

@HOOKS.register_module()
class FreezeParams(Hook):
    def __init__(self, layers=['backbone','neck']):
        self.layers = layers

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        
        for name, param in runner.model.named_parameters():
            for layer in self.layers:
                if layer in name:
                    param.requires_grad = False
            if param.requires_grad:
                print(name)

        return super().after_load_checkpoint(runner, checkpoint)