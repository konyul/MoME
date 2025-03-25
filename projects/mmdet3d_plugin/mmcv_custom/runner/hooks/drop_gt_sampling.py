from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DropGTSamplingHook(Hook):

    def __init__(self,
                 drop_epoch,
                 pipeline_name="UnifiedObjectSample",
                 *args,
                 **kwargs):
        super(DropGTSamplingHook, self).__init__(*args, **kwargs)
        self.drop_epoch = drop_epoch
        self.pipeline_name = pipeline_name
        self.dropped = False

    def before_train_epoch(self, runner) -> None:
        if not self.dropped and runner.epoch >= self.drop_epoch:
            dataset = runner.data_loader.dataset.dataset
            if hasattr(dataset, 'datasets'):
                datasets = dataset.datasets
            else:
                datasets = [dataset]

            for d in datasets:
                pipeline = d.pipeline.transforms
                index = 0
                dropped = False

                for i, p in enumerate(pipeline):
                    if p.__class__.__name__ == self.pipeline_name:
                        index = i
                        dropped = True
                        runner.logger.info(f"{self.pipeline_name} is dropped after {self.drop_epoch} epoch training!")
                        break

                if dropped:
                    pipeline.pop(index)
                    self.dropped = dropped
