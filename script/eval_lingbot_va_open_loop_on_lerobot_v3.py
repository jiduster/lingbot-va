#!/usr/bin/env python
"""Open-loop evaluation on a local LeRobot v3 dataset.

This script reuses the LingBot-VA server stack, loads a transformer checkpoint
override, rolls out a chosen training episode from its first observation, and
writes a comparison video plus a small metrics json.
"""

import argparse
import importlib.util
import json
import math
import os
import time
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video


DEFAULT_DATASET_ROOT = "/mnt/ceph3/zyh/lerobot_pidata"
DEFAULT_BASE_MODEL_ROOT = "/mnt/ceph/ckpt/lingbot-va-posttrain-robotwin"
WAN_VA_SERVER_PATH = Path(__file__).resolve().parents[1] / "wan_va" / "wan_va_server.py"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_csv(value):
    if value is None:
        return None
    values = [item.strip() for item in value.split(",")]
    return [item for item in values if item]


def parse_episode_selector(value):
    if not value:
        return None
    selected = set()
    for part in parse_csv(value):
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return selected


def resolve_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resize_resample_filter():
    if hasattr(cv2, "INTER_CUBIC"):
        return cv2.INTER_CUBIC
    return cv2.INTER_LINEAR


def get_camera_keys(info, requested):
    if requested:
        return requested
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "image"
    ]


def get_episode_chunk(row, info):
    chunks_size = int(info.get("chunks_size", 1000))
    episode_index = int(row["episode_index"])
    return int(row.get("data/chunk_index", episode_index // chunks_size))


def resolve_data_file(root, info, row):
    episode_index = int(row["episode_index"])
    chunk_index = get_episode_chunk(row, info)
    file_index = int(row.get("data/file_index", episode_index))
    rel_path = info["data_path"].format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return root / rel_path


def decode_image_entry(root, parquet_path, entry):
    payload = None
    image_path = None

    if isinstance(entry, dict):
        payload = entry.get("bytes")
        image_path = entry.get("path")
    elif isinstance(entry, (bytes, bytearray, memoryview)):
        payload = entry

    if payload is not None:
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image bytes in {parquet_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    if not image_path:
        raise ValueError(f"Image entry in {parquet_path} has no bytes or path")

    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = root / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing image file: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_frame(root, parquet_path, camera_key, frame_index):
    table = pq.read_table(parquet_path, columns=["frame_index", camera_key])
    rows = table.to_pylist()
    images_by_frame = {int(row["frame_index"]): row[camera_key] for row in rows}
    if frame_index not in images_by_frame:
        raise KeyError(f"Missing frame {frame_index} in {parquet_path}")
    return decode_image_entry(root, parquet_path, images_by_frame[frame_index])


def load_episode_rows(root):
    jsonl_path = root / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        return sorted(read_jsonl(jsonl_path), key=lambda row: row["episode_index"])

    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    wanted_columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    rows = []
    for path in episode_files:
        names = pq.ParquetFile(path).schema_arrow.names
        columns = [name for name in wanted_columns if name in names]
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return sorted(rows, key=lambda row: row["episode_index"])


def find_latent_path(root, camera_key, episode_chunk, episode_index, start_frame, end_frame):
    return (
        root
        / "latents"
        / f"chunk-{episode_chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    )


def load_combined_latent_video(root, camera_keys, episode_chunk, episode_index, start_frame, end_frame):
    latents = []
    latent_paths = []
    base_meta = None
    for camera_key in camera_keys:
        latent_path = find_latent_path(
            root,
            camera_key,
            episode_chunk,
            episode_index,
            start_frame,
            end_frame,
        )
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent file: {latent_path}")
        latent_dict = torch.load(latent_path, map_location="cpu", weights_only=False)
        latent = reconstruct_latent_tensor(latent_dict)
        latents.append(latent)
        latent_paths.append(str(latent_path))
        if base_meta is None:
            base_meta = latent_dict
    combined = torch.cat(latents, dim=3)
    return combined, base_meta, latent_paths


def reconstruct_latent_tensor(latent_dict):
    latent = latent_dict["latent"]
    if not torch.is_tensor(latent):
        latent = torch.tensor(latent)
    latent = latent.to(torch.float32)
    channels = latent.shape[-1]
    latent = latent.view(
        latent_dict["latent_num_frames"],
        latent_dict["latent_height"],
        latent_dict["latent_width"],
        channels,
    ).permute(3, 0, 1, 2).contiguous()
    return latent


def add_title_bar(img, text, font_scale=0.7, thickness=2):
    h, w, _ = img.shape
    bar_height = 36
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = max(0, (w - text_w) // 2)
    text_y = (bar_height + text_h) // 2 - 4
    cv2.putText(
        title_bar,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, img])


def make_comparison_frame(real_frames, pred_frames, idx):
    real_img = real_frames[min(idx, len(real_frames) - 1)]
    pred_img = pred_frames[min(idx, len(pred_frames) - 1)]

    if pred_img.shape[:2] != real_img.shape[:2]:
        pred_img = cv2.resize(pred_img, (real_img.shape[1], real_img.shape[0]))

    row = np.hstack([real_img, pred_img])
    row = add_title_bar(row, f"frame {idx} | real / predicted")
    return row


def decode_action_tensor(action_tensor):
    action = action_tensor.detach().cpu().float()
    if action.ndim == 5:
        action = action[0]
    if action.ndim == 4:
        action = action[..., 0]
    return action.numpy()


def load_first_obs(root, info, row, camera_keys, frame_id, height, width):
    parquet_path = resolve_data_file(root, info, row)
    obs = {}
    for camera_key in camera_keys:
        img = load_frame(root, parquet_path, camera_key, int(frame_id))
        img = cv2.resize(img, (width, height), interpolation=resize_resample_filter())
        obs[camera_key] = img
    return {"obs": [obs]}


def select_output_frame_ids(latent_num_frames, frame_chunk_size, max_chunks):
    num_chunks = min(max_chunks, int(math.ceil(latent_num_frames / frame_chunk_size)))
    return num_chunks


def get_distributed_context(args):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", str(parse_local_rank(args.device))))
    return rank, local_rank, world_size


def maybe_init_distributed(rank, local_rank, world_size):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        kwargs = dict(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        if torch.cuda.is_available():
            kwargs["device_id"] = local_rank
        try:
            dist.init_process_group(**kwargs)
        except TypeError:
            kwargs.pop("device_id", None)
            dist.init_process_group(**kwargs)


def is_main_process(rank):
    return rank == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def parse_local_rank(device):
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    text = str(device).strip()
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    if text.isdigit():
        return int(text)
    return 0


def load_va_server_class():
    if "configs" not in sys.modules:
        fake_configs = types.ModuleType("configs")
        fake_configs.VA_CONFIGS = {}
        sys.modules["configs"] = fake_configs

    spec = importlib.util.spec_from_file_location("lingbot_va_server_eval", WAN_VA_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load VA_Server from {WAN_VA_SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.VA_Server


@torch.no_grad()
def decode_latents_in_chunks(server, latents, output_type, chunk_size):
    chunks = []
    with torch.inference_mode():
        for start in range(0, latents.shape[2], chunk_size):
            chunk = latents[:, :, start : start + chunk_size].detach()
            chunks.append(server.decode_one_video(chunk, output_type)[0])
            torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0) if chunks else None


def move_vae_for_decode(server):
    if server.enable_offload:
        server.vae = server.vae.to(server.device).to(server.dtype)


def offload_vae_after_decode(server):
    if server.enable_offload:
        server.vae = server.vae.to("cpu")
        torch.cuda.empty_cache()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_pika_eval_config(camera_keys, args, rank, local_rank, world_size):
    return types.SimpleNamespace(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        param_dtype=resolve_dtype(args.dtype),
        save_root=str(Path(args.output_root) / f"_rank_{rank}"),
        enable_offload=True,
        infer_mode="i2va",
        wan22_pretrained_model_name_or_path=args.base_model_root,
        transformer_checkpoint_path=args.checkpoint_path,
        attn_window=30,
        frame_chunk_size=4,
        patch_size=(1, 2, 2),
        env_type="none",
        height=args.height,
        width=args.width,
        action_dim=30,
        action_per_frame=16,
        obs_cam_keys=camera_keys,
        guidance_scale=5,
        action_guidance_scale=1,
        num_inference_steps=5,
        video_exec_step=-1,
        action_num_inference_steps=10,
        snr_shift=5.0,
        action_snr_shift=1.0,
        used_action_channel_ids=list(range(14, 20)) + list(range(21, 27)) + [28, 29],
        action_norm_method="quantiles",
        norm_stat={
            "q01": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.7952988283168084,
                0.01894931887623596,
                -1.4064310307234102,
                -0.2614144469771252,
                -0.11770194387874143,
                -0.14157598820857048,
                0.0,
                -0.1302874405557562,
                0.005391807570955586,
                -1.2624255755755804,
                -0.38438282416816666,
                -0.2097004488009648,
                -0.6389200688468439,
                0.0,
                0.061852886875466216,
                0.061779203675131164,
            ],
            "q99": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.011398870030851407,
                1.962559677346168,
                -0.01420569682169639,
                0.22437823589702593,
                0.5978270396427998,
                0.4522191492195597,
                1.0,
                0.6827524886066715,
                1.9211535339352235,
                -0.019798142936269417,
                0.366206134268609,
                0.6604384025425191,
                0.2617562958323477,
                1.0,
                0.09601863311065748,
                0.09644745793556762,
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Open-loop evaluation on local LeRobot v3 data.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--base-model-root", default=DEFAULT_BASE_MODEL_ROOT)
    parser.add_argument("--checkpoint-path", default=None, required=True)
    parser.add_argument("--config-name", default="pika")
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--camera-keys", default=None)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--output-root", default="./eval_out/pika_open_loop")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gt-decode-chunk-size", type=int, default=4)
    parser.add_argument("--prompt-from", choices=["episode", "latent"], default="episode")
    args = parser.parse_args()

    rank, local_rank, world_size = get_distributed_context(args)
    maybe_init_distributed(rank, local_rank, world_size)
    main_process = is_main_process(rank)

    root = Path(args.dataset_root)
    info = read_json(root / "meta" / "info.json")
    episode_rows = load_episode_rows(root)
    selected = parse_episode_selector(args.episodes)
    if selected is not None:
        episode_rows = [row for row in episode_rows if int(row["episode_index"]) in selected]
    episode_rows = episode_rows[: args.max_episodes]
    if not episode_rows:
        raise ValueError("No episodes selected")

    camera_keys = get_camera_keys(info, parse_csv(args.camera_keys))
    if not camera_keys:
        raise ValueError("No camera keys found")

    config = build_pika_eval_config(camera_keys, args, rank, local_rank, world_size)
    VA_Server = load_va_server_class()
    server = VA_Server(config)
    server.video_processor = None
    from diffusers.video_processor import VideoProcessor
    server.video_processor = VideoProcessor(vae_scale_factor=1)

    out_root = Path(args.output_root)
    if main_process:
        out_root.mkdir(parents=True, exist_ok=True)
    barrier()

    metrics = []
    for row in episode_rows:
        t_episode = time.perf_counter()
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        episode_chunk = get_episode_chunk(row, info)
        start_frame = int(row.get("action_config", [{}])[0].get("start_frame", 0))
        end_frame = int(row.get("action_config", [{}])[0].get("end_frame", length))
        latent_tensor, latent_dict, latent_paths = load_combined_latent_video(
            root,
            camera_keys,
            episode_chunk,
            episode_index,
            start_frame,
            end_frame,
        )
        if main_process:
            move_vae_for_decode(server)
            t_gt = time.perf_counter()
            gt_video = decode_latents_in_chunks(
                server,
                latent_tensor.unsqueeze(0).to(server.device),
                "np",
                args.gt_decode_chunk_size,
            )
            offload_vae_after_decode(server)
            if main_process:
                print(f"[timing] episode {episode_index} gt_decode: {time.perf_counter() - t_gt:.3f}s")
        else:
            gt_video = None
        barrier()

        prompt = latent_dict["text"] if args.prompt_from == "latent" else row["tasks"][0]
        first_frame_id = int(latent_dict["frame_ids"][0])
        init_obs = load_first_obs(
            root,
            info,
            row,
            camera_keys,
            first_frame_id,
            args.height,
            args.width,
        )

        server._reset(prompt=prompt)
        pred_chunks = []
        for chunk_id in range(select_output_frame_ids(latent_dict["latent_num_frames"], server.job_config.frame_chunk_size, args.max_chunks)):
            frame_st_id = chunk_id * server.job_config.frame_chunk_size
            obs = init_obs if chunk_id == 0 else {}
            t_infer = time.perf_counter()
            action, latents = server._infer(obs, frame_st_id=frame_st_id)
            if main_process:
                print(f"[timing] episode {episode_index} chunk {chunk_id} infer: {time.perf_counter() - t_infer:.3f}s")
            if main_process:
                t_decode = time.perf_counter()
                move_vae_for_decode(server)
                with torch.inference_mode():
                    pred_chunks.append(server.decode_one_video(latents.detach(), "np")[0])
                offload_vae_after_decode(server)
                print(f"[timing] episode {episode_index} chunk {chunk_id} pred_decode: {time.perf_counter() - t_decode:.3f}s")
                torch.cuda.empty_cache()
            barrier()

        if not main_process:
            continue

        pred_video = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros_like(gt_video[:1])
        min_len = min(len(gt_video), len(pred_video))
        gt_trim = gt_video[:min_len]
        pred_trim = pred_video[:min_len]

        concat_frames = [make_comparison_frame(gt_trim, pred_trim, i) for i in range(min_len)]
        episode_dir = out_root / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        export_to_video(concat_frames, str(episode_dir / "comparison.mp4"), fps=10)
        export_to_video(pred_trim, str(episode_dir / "pred.mp4"), fps=10)
        export_to_video(gt_trim, str(episode_dir / "gt.mp4"), fps=10)

        latent_mse = float(np.mean((pred_trim.astype(np.float32) - gt_trim.astype(np.float32)) ** 2))
        metrics.append({
            "episode_index": episode_index,
            "length": length,
            "prompt": prompt,
            "latent_paths": latent_paths,
            "comparison_video": str(episode_dir / "comparison.mp4"),
            "latent_mse": latent_mse,
            "pred_chunks": len(pred_chunks),
        })
        if main_process:
            print(f"[timing] episode {episode_index} total: {time.perf_counter() - t_episode:.3f}s")

    if main_process:
        with open(out_root / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    cleanup_distributed()


if __name__ == "__main__":
    main()
