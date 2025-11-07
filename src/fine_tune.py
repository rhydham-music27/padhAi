from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer, setup_chat_format
import evaluate

logger = logging.getLogger(__name__)

# Env-driven defaults
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
TRAIN_DATA_PATH = Path(os.getenv("TRAIN_DATA_PATH", "data/processed/combined_train.jsonl"))
VAL_DATA_PATH = Path(os.getenv("VAL_DATA_PATH", "data/processed/combined_val.jsonl"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "models/adapters"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", "3"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
USE_QLORA = os.getenv("USE_QLORA", "true").lower() in ("true", "1", "yes")
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "500"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "500"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "10"))


def create_bnb_config(use_qlora: bool = True) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytesConfig for 4-bit QLoRA or return None for full precision."""
    if not use_qlora:
        logger.info("QLoRA disabled; loading model in full precision.")
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    logger.info("Using QLoRA 4-bit with NF4, double quant, compute_dtype=%s", compute_dtype)
    return bnb_config


def create_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """Create LoRA configuration for CAUSAL_LM on Mistral (attn + MLP projections)."""
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    logger.info("LoRA config: r=%d alpha=%d dropout=%.3f targets=%s", r, alpha, dropout, target_modules)
    return cfg


def load_datasets(train_path: Path, val_path: Path) -> Tuple[Any, Any]:
    """Load train/validation JSONL datasets with a required 'messages' column."""
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing dataset file(s): {train_path} or {val_path}")
    try:
        ds_dict = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(val_path)},
        )
        train_ds, val_ds = ds_dict["train"], ds_dict["validation"]
        for split_name, split in ("train", train_ds), ("validation", val_ds):
            if "messages" not in split.column_names:
                raise ValueError(f"Dataset split '{split_name}' lacks 'messages' column")
        logger.info("Loaded datasets: train=%d rows, val=%d rows", len(train_ds), len(val_ds))
        return train_ds, val_ds
    except Exception as e:
        logger.exception("Failed to load datasets: %s", e)
        raise


def load_model_and_tokenizer(
    model_name: str, bnb_config: Optional[BitsAndBytesConfig]
) -> Tuple[Any, Any]:
    """Load base model/tokenizer with optional 4-bit config and setup chat format."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=(torch.bfloat16 if bnb_config else "auto"),
        )
        if bnb_config:
            model = prepare_model_for_kbit_training(model)
        model, tokenizer = setup_chat_format(model, tokenizer)
        logger.info("Model loaded: %s | 4-bit=%s", model_name, bool(bnb_config))
        return model, tokenizer
    except Exception as e:
        logger.exception("Failed to load model/tokenizer: %s", e)
        raise


def create_training_arguments(
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    max_seq_len: int,
) -> SFTConfig:
    """Create TRL SFTConfig used by SFTTrainer."""
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=max_seq_len,
        packing=False,
        dataset_text_field=None,
        predict_with_generate=True,
        report_to=["tensorboard"],
    )
    try:
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        n_gpu = 1
    eff_bs = batch_size * grad_accum * max(1, n_gpu)
    logger.info("Training args created. Effective batch size=%d", eff_bs)
    return args

def compute_metrics_factory(tokenizer: Any):
    """Create a compute_metrics callback to compute BLEU and ROUGE during eval."""
    def _compute(eval_pred) -> Dict[str, float]:
        try:
            predictions = eval_pred.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            label_ids = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred.label_ids
            bleu_metric = evaluate.load("sacrebleu")
            rouge_metric = evaluate.load("rouge")
            pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            bleu = bleu_metric.compute(predictions=pred_texts, references=[[t] for t in ref_texts])
            rouge = rouge_metric.compute(predictions=pred_texts, references=ref_texts)
            return {
                "bleu": float(bleu.get("score", 0.0)),
                "rouge_l": float(rouge.get("rougeL", 0.0)),
            }
        except Exception as e:
            logger.warning("compute_metrics failed: %s", e)
            return {}
    return _compute


def _compute_bleu_rouge(tokenizer, predictions, labels) -> Dict[str, float]:
    try:
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # sacrebleu expects list[str] and references as list[list[str]]
        bleu = bleu_metric.compute(predictions=pred_texts, references=[[t] for t in ref_texts])
        rouge = rouge_metric.compute(predictions=pred_texts, references=ref_texts)
        return {
            "bleu": float(bleu.get("score", 0.0)),
            "rouge_l": float(rouge.get("rougeL", 0.0)),
        }
    except Exception as e:
        logger.warning("Metric computation failed: %s", e)
        return {}


def train_model(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    lora_config: LoraConfig,
    training_args: SFTConfig,
) -> SFTTrainer:
    """Train the model with SFTTrainer. Returns the trainer instance."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
        compute_metrics=compute_metrics_factory(tokenizer),
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")
    return trainer


def save_model(trainer: SFTTrainer, output_dir: Path, save_merged: bool = False) -> None:
    """Save LoRA adapters and optionally merged model."""
    out_final = output_dir / "final"
    out_final.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_final)
    trainer.tokenizer.save_pretrained(out_final)

    metadata = {
        "model_name": BASE_MODEL_NAME,
        "training_args": json.loads(trainer.args.to_json_string()),
        "output_dir": str(out_final),
    }
    (out_final / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Best-effort to save eval metrics if available
    try:
        metrics_path = out_final / "evaluation_metrics.json"
        if hasattr(trainer, "state") and getattr(trainer.state, "log_history", None):
            last_eval = [m for m in trainer.state.log_history if "eval_loss" in m]
            if last_eval:
                metrics_path.write_text(json.dumps(last_eval[-1], indent=2))
    except Exception:
        pass

    if save_merged:
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        try:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            trainer.tokenizer.save_pretrained(merged_dir)
            logger.warning("Merged model saved (~14GB): %s", merged_dir)
        except Exception as e:
            logger.exception("Failed to merge and save model: %s", e)
            raise


def evaluate_model(trainer: SFTTrainer, test_dataset: Optional[Any] = None) -> Dict[str, float]:
    """Evaluate on validation/test set and return metrics dict."""
    metrics = trainer.evaluate(eval_dataset=test_dataset or trainer.eval_dataset)
    try:
        if "eval_loss" in metrics:
            metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])) .item())
    except Exception:
        pass
    out_final = Path(trainer.args.output_dir) / "final"
    out_final.mkdir(parents=True, exist_ok=True)
    (out_final / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral 7B with LoRA/QLoRA for educational explanations",
    )
    parser.add_argument("--model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--val-data", type=Path, default=VAL_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--use-qlora", action="store_true", default=USE_QLORA)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--logging-steps", type=int, default=LOGGING_STEPS)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    try:
        bnb_config = create_bnb_config(args.use_qlora and not args.no_qlora)
        lora_cfg = create_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
        train_ds, val_ds = load_datasets(args.train_data, args.val_data)
        model, tokenizer = load_model_and_tokenizer(args.model_name, bnb_config)
        sft_args = create_training_arguments(
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.learning_rate,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            max_seq_len=args.max_seq_length,
        )
        trainer = train_model(model, tokenizer, train_ds, val_ds, lora_cfg, sft_args)
        save_model(trainer, args.output_dir, args.save_merged)
        metrics = evaluate_model(trainer)
        logger.info("Final metrics: %s", metrics)
    except torch.cuda.OutOfMemoryError:
        logger.exception("CUDA OOM. Try reducing --batch-size or --max-seq-length, and enable --use-qlora.")
        raise
    except Exception as e:
        logger.exception("Fine-tuning failed: %s", e)
        raise


if __name__ == "__main__":
    main()
