# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from datetime import datetime
import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import numpy as np
import sys
from pathlib import Path
from qwenvl.data.data_qwen import IGNORE_INDEX

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None

class HFSaverCallback(TrainerCallback):
    def __init__(self, tokenizer, image_processor, root_dir: pathlib.Path):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.root_dir = root_dir
        self.trainer = None          # will be set once on_train_begin

    # 1️⃣ grab the trainer reference when training starts
    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer = kwargs["trainer"]          # never None here
        return control

    # 2️⃣ save HF-ready checkpoint every save event
    def on_save(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):        # Rank-0 only
            return control

        model = self.trainer.model
        # ZeRO-3: gather full weights
        full_state = self.trainer.accelerator.get_state_dict(model)

        ckpt = self.root_dir / f"step-{state.global_step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt, state_dict=full_state)

        # save tokenizer / processor only the first time
        if not (ckpt / "tokenizer_config.json").exists():
            self.tokenizer.save_pretrained(ckpt)
            self.image_processor.save_pretrained(ckpt)

        logging.info(f"[HF-Saver] wrote HF checkpoint to {ckpt}")
        return control


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    hf_ckpt_root = pathlib.Path(training_args.output_dir) / "hf_checkpoints"
    hf_ckpt_root.mkdir(parents=True, exist_ok=True)
    hf_saver_cb = HFSaverCallback(tokenizer, data_args.image_processor, hf_ckpt_root)

    # ---------------- Metric: exact match on ANSWER: 0/1 ----------------
    def _normalize(txt: str) -> str:
        return txt.strip().replace("\n", " ").upper()

    eval_examples = data_module.get('eval_dataset', None)

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        # logits: (batch, seq_len, vocab)
        pred_ids = logits.argmax(-1)
        # ensure tensors for dim/shape handling
        if isinstance(pred_ids, np.ndarray):
            pred_ids = torch.tensor(pred_ids, device=model.device)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, device=model.device)
        # If dim mismatch from padding concat, reshape to labels
        if pred_ids.dim() == 1 and labels.dim() == 2:
            pred_ids = pred_ids.view_as(labels)
        correct = 0
        show_samples = 10
        samples_printed = 0
        batch_size = labels.shape[0]
        for idx in range(batch_size):
            label_row = labels[idx]
            mask = label_row != IGNORE_INDEX
            gt_tokens = label_row[mask]
            pred_tokens = pred_ids[idx][mask]
            gt_text = tokenizer.decode(gt_tokens, skip_special_tokens=True)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            if samples_printed < show_samples:
                question = "<unknown>"
                if eval_examples is not None and idx < len(eval_examples):
                    q_ids = eval_examples[idx]["input_ids"][0].tolist()
                    question = tokenizer.decode([t for t in q_ids if t != tokenizer.pad_token_id], skip_special_tokens=True)[:120]
                logging.info(f"[Eval sample {idx}]\nQ: {question}\nExpected: {gt_text}\nPredicted: {pred_text}\n")
                samples_printed += 1
            if _normalize(gt_text) == _normalize(pred_text):
                correct += 1
        return {"exact_match": correct / batch_size}


    class ArgmaxTrainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad():
                loss, logits, labels = super().prediction_step(model, inputs, False, ignore_keys)
            # Convert logits to predicted token IDs (same shape as labels)
            preds = logits.argmax(-1)
            return (loss, preds, labels)

    # ensure evaluation does not accumulate huge GPU tensors
    if training_args.eval_accumulation_steps is None:
        training_args.eval_accumulation_steps = 1

    trainer = ArgmaxTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        # callbacks=[hf_saver_cb],
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
