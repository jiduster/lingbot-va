#!/usr/bin/env python
"""Prepare LeRobot v3 metadata for LingBot-VA post-training.

This script keeps the original parquet data untouched. It can write the v2.1
jsonl metadata files that LingBot-VA expects, adding one full-episode
action_config segment per episode by default.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_ROOT = "/mnt/ceph3/zyh/lerobot_pidata"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_tasks(root):
    jsonl_path = root / "meta" / "tasks.jsonl"
    if jsonl_path.exists():
        rows = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return sorted(rows, key=lambda row: row["task_index"])
    return sorted(
        pq.read_table(root / "meta" / "tasks.parquet").to_pylist(),
        key=lambda row: row["task_index"],
    )


def read_episode_rows(root):
    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No v3 episode parquet files found under {root / 'meta' / 'episodes'}")

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


def make_action_config(row, fallback_task):
    tasks = row.get("tasks") or ([fallback_task] if fallback_task else [])
    if isinstance(tasks, str):
        tasks = [tasks]
    action_text = tasks[0] if tasks else fallback_task
    return [{
        "start_frame": 0,
        "end_frame": int(row["length"]),
        "action_text": action_text,
    }]


def make_episode_jsonl_rows(rows, fallback_task):
    out = []
    for row in rows:
        tasks = row.get("tasks") or ([fallback_task] if fallback_task else [])
        if isinstance(tasks, str):
            tasks = [tasks]
        out.append({
            "episode_index": int(row["episode_index"]),
            "tasks": tasks,
            "length": int(row["length"]),
            "action_config": make_action_config(row, fallback_task),
        })
    return out


def has_required_latent_tree(root, episodes, camera_keys):
    missing = []
    for row in episodes:
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        chunk = int(row.get("data/chunk_index", ep_idx // 1000))
        for key in camera_keys:
            path = (
                root
                / "latents"
                / f"chunk-{chunk:03d}"
                / key
                / f"episode_{ep_idx:06d}_0_{length}.pth"
            )
            if not path.exists():
                missing.append(str(path))
                if len(missing) >= 5:
                    return missing
    return missing


def main():
    parser = argparse.ArgumentParser(description="Adapt local LeRobot v3 metadata for LingBot-VA.")
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT, help="Path to the LeRobot v3 dataset root.")
    parser.add_argument("--write", action="store_true", help="Write meta/tasks.jsonl and meta/episodes.jsonl.")
    parser.add_argument(
        "--check-latents",
        action="store_true",
        help="Check for LingBot-VA latent files named episode_{idx}_0_{length}.pth.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    info = read_json(info_path)
    version = str(info.get("codebase_version", ""))
    tasks = read_tasks(root)
    episodes = read_episode_rows(root)
    fallback_task = tasks[0]["task"] if tasks else ""
    episode_jsonl_rows = make_episode_jsonl_rows(episodes, fallback_task)

    print(f"dataset_root: {root}")
    print(f"codebase_version: {version}")
    print(f"episodes: {len(episode_jsonl_rows)}")
    print(f"frames: {sum(row['length'] for row in episode_jsonl_rows)}")
    print(f"tasks: {len(tasks)}")
    print(f"first_episode: {episode_jsonl_rows[0] if episode_jsonl_rows else None}")

    image_keys = [
        key for key, feature in info.get("features", {}).items()
        if feature.get("dtype") in {"image", "video"}
    ]
    print(f"camera_keys: {image_keys}")

    if args.check_latents:
        missing = has_required_latent_tree(root, episode_jsonl_rows, image_keys)
        if missing:
            print("latents: missing")
            for path in missing:
                print(f"  {path}")
        else:
            print("latents: ok")

    if args.write:
        write_jsonl(root / "meta" / "tasks.jsonl", tasks)
        write_jsonl(root / "meta" / "episodes.jsonl", episode_jsonl_rows)
        print("wrote: meta/tasks.jsonl")
        print("wrote: meta/episodes.jsonl")
    else:
        print("dry_run: pass --write to create meta/tasks.jsonl and meta/episodes.jsonl")


if __name__ == "__main__":
    main()
