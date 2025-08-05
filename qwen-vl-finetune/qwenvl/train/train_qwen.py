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
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Seq2SeqTrainer

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
        preds, labels = eval_preds
        labels = labels.copy()
        labels[labels == IGNORE_INDEX] = tokenizer.pad_token_id
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        correct = 0
        for i, (p, g) in enumerate(zip(decoded_preds, decoded_labels)):
            if i < 10:  # print first 10 examples
                question = "<unknown>"
                if eval_examples is not None and i < len(eval_examples):
                    q_ids = eval_examples[i]['input_ids'][0].tolist()
                    question = tokenizer.decode([t for t in q_ids if t != tokenizer.pad_token_id], skip_special_tokens=True)[:120]
                logging.info(f"[Eval sample {i}]\nQ: {question}\nExpected: {g}\nPredicted: {p}\n")
            if _normalize(p) == _normalize(g):
                correct += 1
        return {"exact_match": correct / len(decoded_preds)}

    # Transformers <4.37 do not add generation_config to TrainingArguments
    # ensure older Transformers have generation attributes expected by Seq2SeqTrainer
    if not hasattr(training_args, "generation_config"):
        training_args.generation_config = None
    if not hasattr(training_args, "generation_num_beams"):
        training_args.generation_num_beams = 1
    # ensure attribute exists for older HF even if None
    if not hasattr(training_args, "generation_max_length"):
        training_args.generation_max_length = None
    # limit only new tokens to 8; keeps input length free
    if not hasattr(training_args, "generation_max_new_tokens"):
        training_args.generation_max_new_tokens = 8
    # older versions need this flag for Seq2Seq evaluate path
    if not hasattr(training_args, "predict_with_generate"):
        training_args.predict_with_generate = True
    # ensure max_length, if present, is not smaller than input length
    gml = getattr(training_args, "generation_max_length", None)
    if gml is None:
        training_args.generation_max_length = training_args.model_max_length
    else:
        training_args.generation_max_length = max(gml, training_args.model_max_length)

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        # callbacks=[hf_saver_cb],
        **data_module
    )
    trainer.predict_with_generate = True
    trainer.generation_max_length = 8

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
