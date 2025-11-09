#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import re
from typing import Dict, List, Optional, Tuple


PROMPT_TEMPLATE = (
    "Given the task, assign a binary reward (1=robot successfully carried out task in a single attempt, 0=anything else) "
    "in the format: ANSWER: <score> and nothing else.\n"
    "Task: {task}.\n"
    "<video>"
)


def list_immediate_subdirs(parent_dir: str) -> List[str]:
    entries = os.listdir(parent_dir)
    subdirs = []
    for entry in entries:
        abs_path = os.path.join(parent_dir, entry)
        if os.path.isdir(abs_path):
            subdirs.append(abs_path)
    subdirs.sort()
    return subdirs


def iter_json_files(val_dir: str) -> List[str]:
    if not os.path.isdir(val_dir):
        return []
    json_files: List[str] = []
    for root, _, files in os.walk(val_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))
    json_files.sort()
    return json_files


def extract_relative_video_path(
    raw_video_path: str,
    input_root: str,
    dataset_dirname: str,
    episode_id: Optional[int],
    segment_index: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (original_rel_path_from_root, absolute_source_path).
    The original relative path should start with 'data_rollout_ori/'.
    """
    normalized = raw_video_path.replace("\\", "/")
    marker = "data_rollout_ori/"
    marker_idx = normalized.find(marker)
    if marker_idx != -1:
        rel_after_marker = normalized[marker_idx + len(marker) :]
        original_rel = os.path.join("data_rollout_ori", rel_after_marker)
        abs_src = os.path.join(input_root, rel_after_marker)
        return original_rel, abs_src

    # Fallback: construct from known structure
    # <dataset>/videos_ori/val/<episode_id>/<segment_index>.mp4
    if episode_id is not None:
        rel_after_root = os.path.join(
            dataset_dirname, "videos_ori", "val", str(episode_id), f"{segment_index}.mp4"
        )
    else:
        rel_after_root = os.path.join(
            dataset_dirname, "videos_ori", "val", f"{segment_index}.mp4"
        )
    original_rel = os.path.join("data_rollout_ori", rel_after_root)
    abs_src = os.path.join(input_root, rel_after_root)
    return original_rel, abs_src


def load_json(path: str) -> Optional[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_examples(input_root: str) -> List[Dict]:
    """
    Collects one example per video segment from all dataset folders in input_root.
    """
    examples: List[Dict] = []
    dataset_dirs = list_immediate_subdirs(input_root)
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        ann_val_dir = os.path.join(dataset_dir, "annotation_all_skip1", "val")
        json_files = iter_json_files(ann_val_dir)
        for jf in json_files:
            data = load_json(jf)
            if not data:
                continue
            texts = data.get("texts") or []
            task_text = ""
            if isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], str):
                task_text = texts[0].strip()
            episode_id = data.get("episode_id")
            videos = data.get("videos") or []
            # Create one example per video segment
            for seg_idx, vid_item in enumerate(videos):
                raw_path = vid_item.get("video_path")
                if not raw_path or not isinstance(raw_path, str):
                    continue
                original_rel, abs_src = extract_relative_video_path(
                    raw_path, input_root, dataset_name, episode_id, seg_idx
                )
                examples.append(
                    {
                        "dataset_dirname": dataset_name,
                        "task_text": task_text,
                        "episode_id": episode_id,
                        "segment_index": seg_idx,
                        "original_video_path": original_rel,
                        "source_abs_path": abs_src,
                    }
                )
    return examples


def determine_reward_label(dataset_dirname: str) -> int:
    return 1 if "score6" in dataset_dirname else 0


def split_train_eval(
    items: List[Dict], train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split: aim for ~50/50 label distribution in eval set when possible.
    """
    rng = random.Random(seed)
    total = len(items)
    if total == 0:
        return [], []

    # Partition by label
    pos = [it for it in items if it.get("label") == 1]
    neg = [it for it in items if it.get("label") == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    # Determine target sizes
    train_count = int(total * train_ratio)
    eval_target = total - train_count
    if eval_target <= 0:
        # All train
        mixed_train = list(items)
        rng.shuffle(mixed_train)
        return mixed_train, []

    # Desired balanced eval composition
    desired_pos = eval_target // 2
    desired_neg = eval_target - desired_pos

    take_pos = min(desired_pos, len(pos))
    take_neg = min(desired_neg, len(neg))

    remaining = eval_target - (take_pos + take_neg)
    if remaining > 0:
        # Allocate remaining from classes with leftover, preferring the class with more leftover
        leftover_pos = max(0, len(pos) - take_pos)
        leftover_neg = max(0, len(neg) - take_neg)
        # First pass: assign to the class with more leftovers
        while remaining > 0 and (leftover_pos > 0 or leftover_neg > 0):
            if leftover_pos >= leftover_neg and leftover_pos > 0:
                take_pos += 1
                leftover_pos -= 1
                remaining -= 1
            elif leftover_neg > 0:
                take_neg += 1
                leftover_neg -= 1
                remaining -= 1
            else:
                break

    eval_set = pos[:take_pos] + neg[:take_neg]
    train_set = pos[take_pos:] + neg[take_neg:]
    rng.shuffle(train_set)
    return train_set, eval_set


def safe_copy(src: str, dst: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def build_output_entry(
    target_rel_video_path: str,
    task_text: str,
    reward_label: int,
    original_rel_path: Optional[str],
) -> Dict:
    human_prompt = PROMPT_TEMPLATE.format(task=task_text if task_text else "N/A")
    entry = {
        "video": [target_rel_video_path],
        "conversations": [
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": f"ANSWER: {reward_label}"},
        ],
        "meta": {
            "task": task_text if task_text else "N/A",
            "success": reward_label,
            "episode_return": 300.0 if reward_label == 1 else -300.0,
            "progress": 1.0 if reward_label == 1 else 0.0,
            "discrete": 1,
            "label_type": "binary",
        },
    }
    if original_rel_path:
        entry["original_video_path"] = original_rel_path
    return entry


def process(
    input_path: str,
    output_path: str,
) -> None:
    os.makedirs(output_path, exist_ok=True)
    output_videos_dir = os.path.join(output_path, "videos")
    os.makedirs(output_videos_dir, exist_ok=True)

    examples = collect_examples(input_path)
    if not examples:
        print("No examples found. Exiting.")
        return

    # Attach labels based on dataset name
    for ex in examples:
        ex["label"] = determine_reward_label(ex["dataset_dirname"])

    # Split into train/eval
    train_items, eval_items = split_train_eval(examples, train_ratio=0.8, seed=42)

    # Copy videos and build JSON entries
    train_entries: List[Dict] = []
    eval_entries: List[Dict] = []

    # Deterministic ordering for copy/index assignment
    def sort_key(item: Dict) -> Tuple:
        return (
            item.get("dataset_dirname") or "",
            item.get("episode_id") if item.get("episode_id") is not None else -1,
            item.get("segment_index") if item.get("segment_index") is not None else -1,
            item.get("original_video_path") or "",
        )

    train_items_sorted = sorted(train_items, key=sort_key)
    eval_items_sorted = sorted(eval_items, key=sort_key)

    # Train copies
    for idx, ex in enumerate(train_items_sorted, start=1):
        src = ex.get("source_abs_path") or ""
        target_filename = f"train_{idx:06d}.mp4"
        target_rel = os.path.join("videos", target_filename)
        target_abs = os.path.join(output_path, target_rel)
        if not os.path.isfile(src):
            print(f"[WARN] Missing source video: {src}")
            continue
        if not safe_copy(src, target_abs):
            print(f"[WARN] Failed to copy: {src} -> {target_abs}")
            continue
        train_entries.append(
            build_output_entry(
                target_rel,
                ex.get("task_text") or "",
                ex.get("label", 0),
                ex.get("original_video_path"),
            )
        )

    # Eval copies
    for idx, ex in enumerate(eval_items_sorted, start=1):
        src = ex.get("source_abs_path") or ""
        target_filename = f"eval_{idx:06d}.mp4"
        target_rel = os.path.join("videos", target_filename)
        target_abs = os.path.join(output_path, target_rel)
        if not os.path.isfile(src):
            print(f"[WARN] Missing source video: {src}")
            continue
        if not safe_copy(src, target_abs):
            print(f"[WARN] Failed to copy: {src} -> {target_abs}")
            continue
        eval_entries.append(
            build_output_entry(
                target_rel,
                ex.get("task_text") or "",
                ex.get("label", 0),
                ex.get("original_video_path"),
            )
        )

    # Write JSONs
    train_json_path = os.path.join(output_path, "train.json")
    eval_json_path = os.path.join(output_path, "eval.json")
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_entries, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train_entries)} train entries to: {train_json_path}")
    print(f"Wrote {len(eval_entries)} eval entries to: {eval_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate world model reward dataset (binary) from rollout directories."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/Users/tonyhlee/Downloads/data_rollout_ori",
        help="Root path containing multiple dataset folders (read-only).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/Users/tonyhlee/Downloads/world_model_rewards",
        help="Output directory to write train/eval JSONs and copied videos.",
    )
    args = parser.parse_args()
    process(args.input_path, args.output_path)


if __name__ == "__main__":
    main()


