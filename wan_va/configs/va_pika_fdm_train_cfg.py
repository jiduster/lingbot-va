# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_pika_train_cfg import va_pika_train_cfg

va_pika_fdm_train_cfg = EasyDict(__name__='Config: VA pika FDM train')
va_pika_fdm_train_cfg.update(va_pika_train_cfg)

va_pika_fdm_train_cfg.save_root = './train_out/pika_fdm'
va_pika_fdm_train_cfg.wan22_pretrained_model_name_or_path = (
    './train_out/pika/checkpoints/checkpoint_step_22000'
)

# Follow the author's issue comment: for each sequence, randomly compute either
# the FDM or IDM loss; FDM uses coefficient 1 and does not include dynamic loss.
va_pika_fdm_train_cfg.enable_fdm_training = True
va_pika_fdm_train_cfg.fdm_prob = 0.5
va_pika_fdm_train_cfg.fdm_loss_weight = 1.0
va_pika_fdm_train_cfg.idm_loss_weight = 1.0
va_pika_fdm_train_cfg.dyn_loss_weight = 1.0

va_pika_fdm_train_cfg.dataset_path = '/mnt/ceph3/zyh/lerobot_pidata'
va_pika_fdm_train_cfg.empty_emb_path = os.path.join(
    va_pika_fdm_train_cfg.dataset_path, 'empty_emb.pt'
)
