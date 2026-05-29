# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_pika_cfg import va_pika_cfg

va_pika_merged_train_cfg = EasyDict(__name__='Config: VA pika merged train')
va_pika_merged_train_cfg.update(va_pika_cfg)

va_pika_merged_train_cfg.dataset_path = '/mnt/ceph3/zyh/lerobot_pidata_merged_v30'
va_pika_merged_train_cfg.empty_emb_path = os.path.join(
    va_pika_merged_train_cfg.dataset_path, 'empty_emb.pt'
)
va_pika_merged_train_cfg.save_root = './train_out/pika_merged'
va_pika_merged_train_cfg.enable_wandb = False
va_pika_merged_train_cfg.load_worker = 16
va_pika_merged_train_cfg.save_interval = 1000
va_pika_merged_train_cfg.gc_interval = 50
va_pika_merged_train_cfg.cfg_prob = 0.1

va_pika_merged_train_cfg.learning_rate = 1e-5
va_pika_merged_train_cfg.beta1 = 0.9
va_pika_merged_train_cfg.beta2 = 0.95
va_pika_merged_train_cfg.weight_decay = 0.1
va_pika_merged_train_cfg.warmup_steps = 10
va_pika_merged_train_cfg.batch_size = 1
va_pika_merged_train_cfg.gradient_accumulation_steps = 1
va_pika_merged_train_cfg.num_steps = 50000
