#!/usr/bin/env python
"""Extract LingBot-VA Wan2.2 VAE latents from LeRobot v3 parquet images.

LeRobot v3 can store image frames directly inside parquet files instead of
materializing mp4 files. Wan2.2 VAE only needs an ordered video tensor, so this
script decodes the parquet image bytes, samples/resizes frames, encodes them
with Wan2.2 VAE, and writes the .pth files expected by LingBot-VA.
"""

import argparse
import importlib.util
import json
import os
import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_DATASET_ROOT = "/mnt/ceph3/zyh/lerobot_pidata"
DEFAULT_WAN_SRC = "/data/home/zyh/Wan2.2"
DEFAULT_VAE_PATH = "/mnt/ceph2/ckpt/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
DEFAULT_EMPTY_EMB_PATH = "/mnt/ceph3/zyh/pick-n-place-sq-lerobot-v21/empty_emb.pt"


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


def read_parquet_episode_rows(root):
    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        return []

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
    return rows


def normalize_episode_row(row):
    row = dict(row)
    row["episode_index"] = int(row["episode_index"])
    row["length"] = int(row["length"])
    tasks = row.get("tasks") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    row["tasks"] = tasks
    if not row.get("action_config"):
        action_text = tasks[0] if tasks else ""
        row["action_config"] = [{
            "start_frame": 0,
            "end_frame": row["length"],
            "action_text": action_text,
        }]
    for action_cfg in row["action_config"]:
        action_cfg["start_frame"] = int(action_cfg["start_frame"])
        action_cfg["end_frame"] = int(action_cfg["end_frame"])
        action_cfg.setdefault("action_text", tasks[0] if tasks else "")
    return row


def load_episodes(root):
    parquet_rows = {
        int(row["episode_index"]): row
        for row in read_parquet_episode_rows(root)
    }

    jsonl_path = root / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        rows = []
        for json_row in read_jsonl(jsonl_path):
            episode_index = int(json_row["episode_index"])
            merged = dict(parquet_rows.get(episode_index, {}))
            merged.update(json_row)
            rows.append(merged)
    else:
        rows = list(parquet_rows.values())

    if not rows:
        raise FileNotFoundError(
            f"No episode metadata found under {root / 'meta'}"
        )
    return sorted(
        [normalize_episode_row(row) for row in rows],
        key=lambda row: row["episode_index"],
    )


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


def resolve_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def sample_frame_ids(start_frame, end_frame, ori_fps, target_fps):
    if end_frame <= start_frame:
        raise ValueError(f"Invalid frame range: {start_frame}..{end_frame}")
    if target_fps <= 0 or ori_fps <= 0:
        raise ValueError("fps values must be positive")
    stride = max(1, int(round(float(ori_fps) / float(target_fps))))
    effective_fps = int(round(float(ori_fps) / float(stride)))
    frame_ids = np.arange(start_frame, end_frame, stride, dtype=np.int64)
    if frame_ids.size < 2:
        raise ValueError(
            f"Frame range {start_frame}..{end_frame} is too short after sampling"
        )
    return frame_ids, stride, effective_fps


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
        return Image.open(BytesIO(payload)).convert("RGB")

    if not image_path:
        raise ValueError(f"Image entry in {parquet_path} has no bytes or path")

    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = root / image_path
    return Image.open(image_path).convert("RGB")


def resize_resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def load_video_tensor(root, parquet_path, camera_key, frame_ids, height, width):
    table = pq.read_table(parquet_path, columns=["frame_index", camera_key])
    rows = table.to_pylist()
    images_by_frame = {int(row["frame_index"]): row[camera_key] for row in rows}

    frames = []
    source_height = None
    source_width = None
    resample = resize_resample_filter()
    for frame_id in frame_ids.tolist():
        if frame_id not in images_by_frame:
            raise KeyError(f"Missing frame {frame_id} in {parquet_path}")
        image = decode_image_entry(root, parquet_path, images_by_frame[frame_id])
        if source_height is None:
            source_width, source_height = image.size
        image = image.resize((width, height), resample=resample)
        frames.append(np.asarray(image, dtype=np.float32))

    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()
    video = video.div(127.5).sub(1.0)
    return video, source_height, source_width


class ConstantTextEmbedder:
    def __init__(self, tensor):
        if tensor.ndim != 2:
            raise ValueError(f"text_emb must be [L, D], got {tuple(tensor.shape)}")
        self.tensor = tensor.detach().cpu().to(torch.bfloat16)

    def __call__(self, text):
        return self.tensor.clone()

    def empty(self):
        return self.tensor.clone()


class WanT5TextEmbedder:
    def __init__(self, wan_src, checkpoint, tokenizer, device, dtype, text_len):
        modules_dir = Path(wan_src) / "wan" / "modules"
        t5_module = load_wan_module_without_package_init(
            "_wan_noinit.modules.t5",
            modules_dir / "t5.py",
            package_root=Path(wan_src) / "wan",
            load_tokenizers=True,
        )
        T5EncoderModel = t5_module.T5EncoderModel

        self.encoder = T5EncoderModel(
            text_len=text_len,
            dtype=dtype,
            device=torch.device(device),
            checkpoint_path=str(checkpoint),
            tokenizer_path=str(tokenizer),
        )
        self.device = torch.device(device)

    def encode(self, text):
        ids, mask = self.encoder.tokenizer(
            [text],
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        with torch.inference_mode():
            context = self.encoder.model(ids, mask)[0]
        return context.detach().cpu().to(torch.bfloat16)

    def __call__(self, text):
        return self.encode(text)

    def empty(self):
        return self.encode("")


def build_text_embedder(args, root):
    if args.text_embedding_mode == "zero":
        tensor = torch.zeros(args.text_len, args.text_dim, dtype=torch.bfloat16)
        return ConstantTextEmbedder(tensor)

    if args.text_embedding_mode == "empty":
        candidates = []
        if args.empty_emb_path:
            candidates.append(Path(args.empty_emb_path))
        candidates.append(root / "empty_emb.pt")
        for path in candidates:
            if path.exists():
                tensor = torch.load(path, map_location="cpu", weights_only=False)
                return ConstantTextEmbedder(tensor)
        raise FileNotFoundError(
            "No empty embedding found. Pass --empty-emb-path, use "
            "--text-embedding-mode zero for a smoke test, or provide T5 args."
        )

    if args.text_embedding_mode == "t5":
        if not args.t5_checkpoint or not args.t5_tokenizer:
            raise ValueError(
                "--text-embedding-mode t5 requires --t5-checkpoint and --t5-tokenizer"
            )
        return WanT5TextEmbedder(
            wan_src=Path(args.wan_src),
            checkpoint=Path(args.t5_checkpoint),
            tokenizer=Path(args.t5_tokenizer),
            device=args.t5_device,
            dtype=resolve_dtype(args.t5_dtype),
            text_len=args.text_len,
        )

    raise ValueError(f"Unsupported text embedding mode: {args.text_embedding_mode}")


def maybe_write_empty_emb(root, embedder, dry_run):
    out_path = root / "empty_emb.pt"
    if out_path.exists():
        return
    if dry_run:
        print(f"dry_run: would write {out_path}")
        return
    torch.save(embedder.empty(), out_path)
    print(f"wrote: {out_path}")


def load_vae(args, device):
    module = load_wan_module_without_package_init(
        "_wan_noinit.modules.vae2_2",
        Path(args.wan_src) / "wan" / "modules" / "vae2_2.py",
        package_root=Path(args.wan_src) / "wan",
    )
    Wan2_2_VAE = module.Wan2_2_VAE

    return Wan2_2_VAE(
        vae_pth=args.vae_path,
        dtype=resolve_dtype(args.dtype),
        device=device,
    )


def load_wan_module_without_package_init(
    module_name,
    module_path,
    package_root,
    load_tokenizers=False,
):
    """Load a Wan module file without executing Wan's top-level __init__.py."""
    module_path = Path(module_path)
    package_root = Path(package_root)
    modules_root = package_root / "modules"
    package_name = module_name.rsplit(".", 1)[0]
    root_name = package_name.split(".", 1)[0]

    if root_name not in sys.modules:
        root_pkg = types.ModuleType(root_name)
        root_pkg.__path__ = [str(package_root)]
        sys.modules[root_name] = root_pkg

    if package_name not in sys.modules:
        modules_pkg = types.ModuleType(package_name)
        modules_pkg.__path__ = [str(modules_root)]
        sys.modules[package_name] = modules_pkg

    tokenizer_name = f"{package_name}.tokenizers"
    tokenizer_path = modules_root / "tokenizers.py"
    if (
        load_tokenizers
        and tokenizer_name not in sys.modules
        and tokenizer_path.exists()
    ):
        load_python_module(tokenizer_name, tokenizer_path)

    return load_python_module(module_name, module_path)


def load_python_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def encode_latent(vae, video, device):
    with torch.inference_mode():
        latent = vae.encode([video.to(device)])[0]
    if latent.ndim != 4:
        raise ValueError(f"Expected VAE latent [C, F, H, W], got {tuple(latent.shape)}")
    channels, latent_num_frames, latent_height, latent_width = latent.shape
    flat = latent.permute(1, 2, 3, 0).reshape(-1, channels)
    return (
        flat.detach().cpu().to(torch.bfloat16),
        int(latent_num_frames),
        int(latent_height),
        int(latent_width),
    )


def latent_output_path(root, camera_key, episode_chunk, episode_index, start, end):
    return (
        root
        / "latents"
        / f"chunk-{episode_chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start}_{end}.pth"
    )


def build_jobs(root, info, episodes, camera_keys, args):
    jobs = []
    ori_fps = int(info["fps"])
    for row in episodes:
        episode_index = int(row["episode_index"])
        episode_chunk = get_episode_chunk(row, info)
        parquet_path = resolve_data_file(root, info, row)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing episode parquet file: {parquet_path}")

        for action_cfg in row["action_config"]:
            start_frame = int(action_cfg["start_frame"])
            end_frame = int(action_cfg["end_frame"])
            frame_ids, frame_stride, effective_fps = sample_frame_ids(
                start_frame,
                end_frame,
                ori_fps,
                args.target_fps,
            )
            action_text = action_cfg.get("action_text") or (
                row["tasks"][0] if row.get("tasks") else ""
            )
            for camera_key in camera_keys:
                out_path = latent_output_path(
                    root,
                    camera_key,
                    episode_chunk,
                    episode_index,
                    start_frame,
                    end_frame,
                )
                if out_path.exists() and not args.overwrite:
                    continue
                jobs.append({
                    "episode_index": episode_index,
                    "episode_chunk": episode_chunk,
                    "parquet_path": parquet_path,
                    "camera_key": camera_key,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_ids": frame_ids,
                    "frame_stride": frame_stride,
                    "fps": effective_fps,
                    "ori_fps": ori_fps,
                    "text": action_text,
                    "out_path": out_path,
                })
    return jobs


def save_latent(job, video, source_height, source_width, latent_tuple, text_emb, args):
    flat, latent_num_frames, latent_height, latent_width = latent_tuple
    data = {
        "latent": flat,
        "latent_num_frames": latent_num_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "video_num_frames": int(video.shape[1]),
        "video_height": int(args.height),
        "video_width": int(args.width),
        "source_video_height": int(source_height),
        "source_video_width": int(source_width),
        "text_emb": text_emb.detach().cpu().to(torch.bfloat16),
        "text": job["text"],
        "frame_ids": job["frame_ids"],
        "start_frame": int(job["start_frame"]),
        "end_frame": int(job["end_frame"]),
        "fps": int(job["fps"]),
        "ori_fps": int(job["ori_fps"]),
    }
    out_path = job["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, out_path)


def run_dry_run(root, jobs, args):
    print(f"dry_run: pending_jobs={len(jobs)}")
    for job in jobs[:5]:
        print(
            "dry_run: would write "
            f"{job['out_path']} from {job['parquet_path']} "
            f"frames={len(job['frame_ids'])} stride={job['frame_stride']} "
            f"fps={job['fps']}"
        )
    if not jobs:
        return

    job = jobs[0]
    sample_frame_ids = job["frame_ids"][: args.dry_run_frames]
    video, source_height, source_width = load_video_tensor(
        root,
        job["parquet_path"],
        job["camera_key"],
        sample_frame_ids,
        args.height,
        args.width,
    )
    print(
        "dry_run: decoded sample "
        f"camera={job['camera_key']} tensor={tuple(video.shape)} "
        f"source_hw=({source_height}, {source_width}) "
        f"range=[{int(sample_frame_ids[0])}, {int(sample_frame_ids[-1])}]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract LingBot-VA latent .pth files from LeRobot v3 parquet images."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--wan-src", default=DEFAULT_WAN_SRC)
    parser.add_argument("--vae-path", default=DEFAULT_VAE_PATH)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--camera-keys", default=None, help="Comma-separated camera keys.")
    parser.add_argument("--episodes", default=None, help="Episode ids/ranges, e.g. 0,3-5.")
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-frames", type=int, default=4)
    parser.add_argument(
        "--text-embedding-mode",
        choices=["empty", "zero", "t5"],
        default="empty",
        help="Use an existing empty embedding, all-zero embedding, or Wan2.2 T5.",
    )
    parser.add_argument("--empty-emb-path", default=DEFAULT_EMPTY_EMB_PATH)
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--t5-checkpoint", default=None)
    parser.add_argument("--t5-tokenizer", default=None)
    parser.add_argument("--t5-device", default="cuda")
    parser.add_argument("--t5-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("--height and --width must be divisible by 16 for Wan2.2 VAE")
    if not args.dry_run and not Path(args.vae_path).exists():
        raise FileNotFoundError(f"Missing VAE checkpoint: {args.vae_path}")

    info = read_json(info_path)
    episodes = load_episodes(root)
    selected = parse_episode_selector(args.episodes)
    if selected is not None:
        episodes = [row for row in episodes if row["episode_index"] in selected]
    if args.start_episode is not None:
        episodes = [row for row in episodes if row["episode_index"] >= args.start_episode]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    camera_keys = get_camera_keys(info, parse_csv(args.camera_keys))
    if args.max_cameras is not None:
        camera_keys = camera_keys[: args.max_cameras]
    if not camera_keys:
        raise ValueError("No image camera keys found or requested")

    jobs = build_jobs(root, info, episodes, camera_keys, args)
    print(f"dataset_root: {root}")
    print(f"episodes_selected: {len(episodes)}")
    print(f"camera_keys: {camera_keys}")
    print(f"target_size: {args.height}x{args.width}")
    print(f"target_fps: {args.target_fps}")

    if args.dry_run:
        run_dry_run(root, jobs, args)
        return

    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"pending_jobs: {len(jobs)}")
    if not jobs:
        return

    text_embedder = build_text_embedder(args, root)
    maybe_write_empty_emb(root, text_embedder, dry_run=False)
    vae = load_vae(args, device)

    iterator = tqdm(jobs, desc="extract latents") if tqdm else jobs
    text_cache = {}
    for job in iterator:
        text = job["text"]
        if text not in text_cache:
            text_cache[text] = text_embedder(text)
        video, source_height, source_width = load_video_tensor(
            root,
            job["parquet_path"],
            job["camera_key"],
            job["frame_ids"],
            args.height,
            args.width,
        )
        latent_tuple = encode_latent(vae, video, device)
        save_latent(
            job=job,
            video=video,
            source_height=source_height,
            source_width=source_width,
            latent_tuple=latent_tuple,
            text_emb=text_cache[text],
            args=args,
        )


if __name__ == "__main__":
    main()
