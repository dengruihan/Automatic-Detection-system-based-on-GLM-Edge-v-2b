# -*- coding: utf-8 -*-
import os

os.environ["WANDB_DISABLED"] = "true"
import jieba
import dataclasses as dc
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Union
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, Split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftConfig, get_peft_config, get_peft_model
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, NamedSplit
from typing import Optional
from PIL import Image

app = typer.Typer(pretty_exceptions_show_locals=False)

class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = [feature["output_ids"] for feature in features] if "output_ids" in features[0].keys() else None
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                    (max_output_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_output_length - len(feature["output_ids"]))
                if isinstance(feature["output_ids"], list):
                    feature["output_ids"] = feature["output_ids"] + remainder
                else:
                    feature["output_ids"] = np.concatenate([feature["output_ids"], remainder]).astype(np.int64)
        return super().__call__(features, return_tensors)

class Seq2SeqTrainer(_Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys=None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            if self.args.predict_with_generate:
                output_ids = inputs.pop("output_ids", None)

            if "labels" in inputs:
                del inputs["labels"]

            loss, generated_tokens, labels = super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )

            if generated_tokens is not None:
                generated_tokens = generated_tokens[:, inputs["input_ids"].size()[1] :]

            if self.args.predict_with_generate:
                labels = output_ids
            
            del inputs, output_ids

        return loss, generated_tokens, labels

@dc.dataclass
class DataConfig(object):
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }

@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int
    freezeV: bool

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(
            output_dir="./output",
            fp16=True,  # 启用混合精度
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            dataloader_drop_last=True,
            group_by_length=True,  # 按长度分组
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            ddp_backend="nccl",  # 多GPU支持
        )
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = "no"
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                self.training_args.per_device_eval_batch_size or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> "FinetuningConfig":
        training_args = kwargs.get("training_args", None)
        if training_args is not None and not isinstance(training_args, Seq2SeqTrainingArguments):
            gen_config = training_args.get("generation_config")
            if not isinstance(gen_config, GenerationConfig):
                training_args["generation_config"] = GenerationConfig(**gen_config)
            kwargs["training_args"] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get("data_config")
        if not isinstance(data_config, DataConfig):
            kwargs["data_config"] = DataConfig(**data_config)

        peft_config = kwargs.get("peft_config", None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs["peft_config"] = get_peft_config(config_dict=peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "FinetuningConfig":
        path = Path(path)
        parser = yaml.YAML(typ="safe", pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)

def _load_datasets(
    data_dir: str,
    data_format: str,
    data_files: dict[NamedSplit, str],
    num_proc: Optional[int],
) -> DatasetDict:
    if data_format == ".jsonl":
        dataset_dct = load_dataset(
            data_dir,
            data_files=data_files,
            split=None,
            num_proc=num_proc,
        )
    else:
        raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
    return dataset_dct

class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            data_dir,
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
        self,
        split: NamedSplit,
        process_fn: Callable[[dict[str, Any]], dict[str, Any]],
        batched: bool = True,
        remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return
        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
            writer_batch_size=1000,
            batch_size=1000,
        )

def process_batch(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    processor,
    max_input_length: int,
    max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch["messages"]
    batched_input_ids = []
    batched_attention_mask = []
    batched_position_ids = []
    batched_labels = []
    batched_images = []

    # 批处理编码
    encoded = tokenizer.apply_chat_template(
        batched_conv,
        padding="max_length",
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
        add_generation_prompt=False,
    )

    # 确保 input_ids 是二维张量 [batch_size, seq_length]
    input_ids = encoded["input_ids"].squeeze(0)  # 去除多余的 batch 维度
    attention_mask = encoded["attention_mask"].squeeze(0)

    # 遍历每个样本
    for idx in range(len(batched_conv)):
        current_input_ids = input_ids[idx].tolist()
        current_attention_mask = attention_mask[idx].tolist()
        position_ids = list(range(len(current_input_ids)))

        # 添加 EOS
        current_input_ids.append(59253)
        current_attention_mask.append(1)
        position_ids.append(len(position_ids))

        # 动态填充
        padding_length = max(0, (max_input_length + max_output_length) - len(current_input_ids))
        padded_input_ids = [tokenizer.pad_token_id] * padding_length + current_input_ids
        padded_attention_mask = [0] * padding_length + current_attention_mask
        padded_position_ids = [0] * padding_length + position_ids

        # 生成 labels
        labels = []
        for pos in range(len(padded_input_ids)):
            if pos >= len(padded_input_ids) - max_output_length:
                labels.append(padded_input_ids[pos])
            else:
                labels.append(-100)

        # 处理图像
        if batched_conv[idx][0]["content"][0].get("image"):
            image = Image.open(batched_conv[idx][0]["content"][0]["image"])
            pixel_values = torch.tensor(processor(image).pixel_values)
        else:
            pixel_values = torch.zeros([3, 672, 672])

        batched_input_ids.append(padded_input_ids)
        batched_attention_mask.append(padded_attention_mask)
        batched_position_ids.append(padded_position_ids)
        batched_labels.append(labels)
        batched_images.append(pixel_values)

    return {
        "input_ids": batched_input_ids,
        "attention_mask": batched_attention_mask,
        "position_ids": batched_position_ids,
        "labels": batched_labels,
        "pixel_values": batched_images,
    }

def process_batch_eval(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    processor,
    max_input_length: int,
    max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch["messages"]
    batched_input_ids = []
    batched_attention_mask = []
    batched_position_ids = []
    batched_output_ids = []
    batched_images = []

    for conv in batched_conv:
        encoded = tokenizer.apply_chat_template(
            conv,
            padding="max_length",
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        position_ids = list(range(len(input_ids)))

        # 生成输出段
        output_segment = input_ids[-max_output_length:]
        output_segment.append(59253)  # EOS

        # 图像处理
        if conv[0]["content"][0].get("image"):
            image = Image.open(conv[0]["content"][0]["image"])
            pixel_values = torch.tensor(processor(image).pixel_values)
        else:
            pixel_values = torch.zeros([3, 672, 672])

        batched_input_ids.append(input_ids[:max_input_length])
        batched_attention_mask.append(attention_mask[:max_input_length])
        batched_position_ids.append(position_ids[:max_input_length])
        batched_output_ids.append(output_segment[:max_output_length])
        batched_images.append(pixel_values)

    return {
        "input_ids": batched_input_ids,
        "attention_mask": batched_attention_mask,
        "position_ids": batched_position_ids,
        "output_ids": batched_output_ids,
        "pixel_values": batched_images,
    }

def load_tokenizer_and_model(
    model_dir: str,
    peft_config: Optional[PeftConfig] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side="left")
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float16)  # 改为fp16
    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
    return tokenizer, model, processor

@app.command()
def main(
    data_dir: Annotated[str, typer.Argument(help="")],
    model_dir: Annotated[
        str,
        typer.Argument(
            help="A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file."
        ),
    ],
    config_file: Annotated[str, typer.Argument(help="")],
    auto_resume_from_checkpoint: str = typer.Argument(
        default="",
        help="If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training",
    ),
):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model, processor = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)

    if ft_config.freezeV:
        for param in model.base_model.model.model.vision.parameters():
            param.requires_grad = False
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            processor=processor,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )

    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            processor=processor,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )

    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            processor=processor,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )

    # 移除梯度检查点（仅在显存不足时启用）
    # model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 自动恢复逻辑保持不变
    if auto_resume_from_checkpoint.upper() in ("", "NO"):
        trainer.train()
    else:
        output_dir = ft_config.training_args.output_dir
        checkpoint_sn = 0
        if os.path.exists(output_dir):
            dirlist = os.listdir(output_dir)
            for checkpoint_str in dirlist:
                if "checkpoint" in checkpoint_str and "tmp" not in checkpoint_str:
                    current_sn = int(checkpoint_str.split("-")[-1])
                    if current_sn > checkpoint_sn:
                        checkpoint_sn = current_sn
        
        if auto_resume_from_checkpoint.upper() == "YES" and checkpoint_sn > 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{checkpoint_sn}")
            print(f"Resuming from checkpoint: {checkpoint_dir}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        elif auto_resume_from_checkpoint.isdigit():
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{auto_resume_from_checkpoint}")
            if os.path.exists(checkpoint_dir):
                print(f"Resuming from specified checkpoint: {checkpoint_dir}")
                trainer.train(resume_from_checkpoint=checkpoint_dir)
            else:
                print(f"Checkpoint {auto_resume_from_checkpoint} not found. Starting from scratch.")
                trainer.train()
        else:
            trainer.train()

    if test_dataset is not None:
        trainer.predict(test_dataset)

if __name__ == "__main__":
    app()