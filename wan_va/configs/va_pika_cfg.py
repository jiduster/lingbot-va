# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_pika_cfg = EasyDict(__name__='Config: VA pika')
va_pika_cfg.update(va_shared_cfg)

va_pika_cfg.wan22_pretrained_model_name_or_path = "/mnt/ceph/ckpt/lingbot-va-posttrain-robotwin"

va_pika_cfg.attn_window = 30
va_pika_cfg.frame_chunk_size = 4
va_pika_cfg.env_type = 'none'

va_pika_cfg.height = 256
va_pika_cfg.width = 256
va_pika_cfg.action_dim = 30
va_pika_cfg.action_per_frame = 16
va_pika_cfg.obs_cam_keys = [
    'observation.images.scene',
    'observation.images.left_wrist',
    'observation.images.right_wrist',
]
va_pika_cfg.guidance_scale = 5
va_pika_cfg.action_guidance_scale = 1

va_pika_cfg.num_inference_steps = 5
va_pika_cfg.video_exec_step = -1
va_pika_cfg.action_num_inference_steps = 10

va_pika_cfg.snr_shift = 5.0
va_pika_cfg.action_snr_shift = 1.0

# Pika data stores 12 arm joints followed by left/right gripper:
# left joints [0:6], right joints [6:12], left gripper [12], right gripper [13].
# LingBot-VA keeps a 30-d action space:
# left EEF [0:7], right EEF [7:14], left joints [14:21], right joints [21:28],
# left gripper [28], right gripper [29].
va_pika_cfg.used_action_channel_ids = (
    list(range(14, 20)) + list(range(21, 27)) + [28, 29]
)
inverse_used_action_channel_ids = [
    len(va_pika_cfg.used_action_channel_ids)
] * va_pika_cfg.action_dim
for i, j in enumerate(va_pika_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_pika_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_pika_cfg.action_norm_method = 'quantiles'
va_pika_cfg.norm_stat = {
    "q01": [0.] * 14 + [
        -0.7952988283168084,
        0.01894931887623596,
        -1.4064310307234102,
        -0.2614144469771252,
        -0.11770194387874143,
        -0.14157598820857048,
        0.,
        -0.1302874405557562,
        0.005391807570955586,
        -1.2624255755755804,
        -0.38438282416816666,
        -0.2097004488009648,
        -0.6389200688468439,
        0.,
        0.061852886875466216,
        0.061779203675131164,
    ],
    "q99": [0.] * 14 + [
        -0.011398870030851407,
        1.962559677346168,
        -0.01420569682169639,
        0.22437823589702593,
        0.5978270396427998,
        0.4522191492195597,
        1.,
        0.6827524886066715,
        1.9211535339352235,
        -0.019798142936269417,
        0.366206134268609,
        0.6604384025425191,
        0.2617562958323477,
        1.,
        0.09601863311065748,
        0.09644745793556762,
    ],
}
