from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class FreezeWeight(Hook):
    def __init__(self, finetune_weight):
        super().__init__()
        self.finetune_weight = finetune_weight

    def before_run(self, runner):
        if hasattr(runner.model, "module"):
            model = runner.model.module
        else:
            model = runner.model

        freezed = []
        not_freezed = []
        for name, p in model.named_parameters():
            flag = False
            for f in self.finetune_weight:
                if name.startswith(f) and p.requires_grad:
                    flag = True
                    not_freezed.append(name)

            if not flag:
                p.requires_grad = False
                freezed.append(name)

        runner.logger.info(f"Freezed parameters: {', '.join(freezed)}")
        runner.logger.info(f"Learned parameters: {', '.join(not_freezed)}")
