# Copyright (c) 2023 megvii-model. All Rights Reserved.

import importlib
import os
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


class Wrapper:

    def __init__(self,
                 cfg,
                 checkpoint=None) -> None:
        self.cfg = Config.fromfile(cfg)
        self.save_dir = './tmp'
        self.init()
        self.model = self._build_model(checkpoint)

        if self.cfg.get('fp16', None) is not None:
            wrap_fp16_model(self.model)
        if cfg.get('optimizer_config', None) is not None and cfg.optimizer_config['type'] == 'CustomFp16OptimizerHook':
            wrap_fp16_model(self.model)
            for module_name, v in cfg.optimizer_config['custom_fp16'].items():
                self.model._modules[module_name].fp16_enabled = v

        self.dataset = self._build_dataset()

    def init(self):
        self.cfg.model.pretrained = None
        self.cfg.data.test.test_mode = True
        plugin_dir = self.cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]
        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)

    def _build_model(self, checkpoint=None):
        model = build_detector(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        if checkpoint:
            load_checkpoint(model, checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        return model

    def _build_dataset(self):
        dataset = build_dataset(self.cfg.data.val)
        return dataset

    def test_speed(self, num_iters=100):
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        loader = iter(data_loader)
        total_time = 0
        warmup_iter = 10

        with torch.no_grad():
            for _ in range(num_iters):
                data = next(loader)
                t1 = time.time()
                self.model(**data, return_loss=False)

                if _ >= warmup_iter:
                    total_time += time.time() - t1

        print(f'Average time: {total_time / (num_iters - warmup_iter)}')


if __name__ == '__main__':
    wrapper = Wrapper(
        cfg='your path to config file',
    )
    wrapper.test_speed()
