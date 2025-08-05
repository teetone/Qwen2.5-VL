#!/usr/bin/env python3
"""Add missing <video> tag to the first human prompt of every example.

Usage (from project root):
    python tools/add_video_tag.py \
        --src /nlp/scr4/nlp/crfm/text2image/text2image-rlhf/robotics/roboreward/roboreward/logs/reward_data/eval.json \
        --dst /nlp/scr4/nlp/crfm/text2image/text2image-rlhf/robotics/roboreward/roboreward/logs/reward_data/eval_with_video.json

The script keeps the source file intact and writes a new JSON file with the
string "\nCamera:\n<video>" appended to the first (human) conversation turn
of every example.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Dict, Any


ADDITION = "\nCamera:\n<video>"


def _patch_single(example: Dict[str, Any]) -> Dict[str, Any]:
    """Return a *new* dict with the modified first human prompt."""
    conversations: List[Dict[str, str]] = example.get("conversations", [])
    if not conversations:
        return example  # nothing to do

    # Work on a shallow copy to avoid mutating input data accidentally
    conversations = [conv.copy() for conv in conversations]
    first_turn = conversations[0]

    # Only touch the first message if it's a human prompt
    if (first_turn.get("from") or first_turn.get("role")) in {"human", "user"}:
        key = "value" if "value" in first_turn else "content"
        if ADDITION.strip() not in first_turn.get(key, ""):
            first_turn[key] = f"{first_turn.get(key, '')}{ADDITION}"
        conversations[0] = first_turn

    patched = example.copy()
    patched["conversations"] = conversations
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description="Add <video> tag to human prompts.")
    parser.add_argument("--src", required=True, help="Path to the original train.json")
    parser.add_argument("--dst", required=True, help="Path to write the patched JSON array")
    args = parser.parse_args()

    src_path = pathlib.Path(args.src)
    dst_path = pathlib.Path(args.dst)

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Source JSON must be an array of examples")

    patched = [_patch_single(ex) for ex in data]

    # Make sure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(patched, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(patched)} examples to {dst_path}")


if __name__ == "__main__":
    main()
