"""Data preparation utilities for educational datasets.

This module loads ELI5, SciQ, OpenBookQA, and WikiHow-like datasets,
formats them into instruction-tuning JSONL compatible with Mistral 7B
(chat messages), and provides simple augmentation utilities.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"

# System prompt used in messages
SYSTEM_PROMPT = (
    "You are an expert educational tutor specializing in explaining complex concepts to "
    "undergraduate students. Provide clear, accurate, and engaging explanations using "
    "analogies, examples, and step-by-step reasoning when appropriate."
)

# Seed
DEFAULT_SEED = int(os.getenv("RANDOM_SEED", "42"))
random.seed(DEFAULT_SEED)


def _select_max_samples(ds: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is not None and len(ds) > max_samples:
        return ds.select(range(max_samples))
    return ds


def load_eli5_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """Load and preprocess ELI5 dataset.

    Tries canonical loaders in order, returning a flattened Dataset
    with columns: question, context, answer.
    """
    split_map = {
        "train": "train_eli5",
        "validation": "validation_eli5",
        "test": "test_eli5",
    }
    split_name = split_map.get(split, split)

    last_error: Optional[Exception] = None
    # Attempt 1: canonical eli5
    try:
        ds = load_dataset("eli5", split=split_name)
        # answers has fields: text (list), score (list)
        def _best_answer(ex: Dict[str, Any]) -> str:
            texts = ex.get("answers", {}).get("text", [])
            scores = ex.get("answers", {}).get("score", [])
            if texts and scores and len(texts) == len(scores):
                idx = max(range(len(scores)), key=lambda i: scores[i])
                return texts[idx]
            return texts[0] if texts else ""

        ds = ds.map(
            lambda ex: {
                "question": ex.get("title", "").strip(),
                "context": ex.get("selftext", "").strip(),
                "answer": _best_answer(ex).strip(),
            },
            remove_columns=[c for c in ds.column_names if c not in ("title", "selftext", "answers")],
        )
        ds = ds.filter(lambda ex: bool(ex["question"]) and bool(ex["answer"]))
        return _select_max_samples(ds, max_samples)
    except Exception as e:  # noqa: BLE001
        last_error = e
        logger.warning("Primary ELI5 loader failed: %s", e)

    # Attempt 2: sentence-transformers/eli5
    try:
        ds = load_dataset("sentence-transformers/eli5", "pair", split=split)
        # Expect fields: query, passage
        ds = ds.map(
            lambda ex: {
                "question": ex.get("query", "").strip(),
                "context": "",
                "answer": ex.get("passage", "").strip(),
            },
            remove_columns=[c for c in ds.column_names if c not in ("query", "passage")],
        )
        ds = ds.filter(lambda ex: bool(ex["question"]) and bool(ex["answer"]))
        return _select_max_samples(ds, max_samples)
    except Exception as e:  # noqa: BLE001
        last_error = e
        logger.warning("Fallback ELI5 loader (sentence-transformers) failed: %s", e)

    # Attempt 3: rexarski/eli5_category
    try:
        ds = load_dataset("rexarski/eli5_category", split=split)
        ds = ds.map(
            lambda ex: {
                "question": (ex.get("title") or "").strip(),
                "context": (ex.get("selftext") or "").strip(),
                "answer": (ex.get("answer") or "").strip(),
            }
        )
        ds = ds.filter(lambda ex: bool(ex["question"]) and bool(ex["answer"]))
        return _select_max_samples(ds, max_samples)
    except Exception as e:  # noqa: BLE001
        logger.error("All ELI5 loaders failed. Last error: %s", e)
        if last_error is not None:
            raise last_error
        raise


def load_sciq_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """Load and preprocess SciQ dataset.

    Returns dataset with question, answer, support, distractors.
    """
    try:
        ds = load_dataset("allenai/sciq", split=split)
        ds = ds.map(
            lambda ex: {
                "question": (ex.get("question") or "").strip(),
                "answer": (ex.get("correct_answer") or "").strip(),
                "support": (ex.get("support") or "").strip(),
                "distractors": [
                    ex.get("distractor1") or "",
                    ex.get("distractor2") or "",
                    ex.get("distractor3") or "",
                ],
            }
        )
        ds = ds.filter(lambda ex: bool(ex["question"]) and bool(ex["answer"]))
        return _select_max_samples(ds, max_samples)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to load SciQ: %s", e)
        raise


def load_openbookqa_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """Load and preprocess OpenBookQA (main config).

    Returns dataset with question, answer, fact, all_choices.
    """
    try:
        ds = load_dataset("allenai/openbookqa", "main", split=split)
        # choices: {label: [A,B,C,D], text: [..]}; answerKey in {A,B,C,D}; fact1 string
        def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
            choices = ex.get("choices", {})
            labels: List[str] = list(choices.get("label") or [])
            texts: List[str] = list(choices.get("text") or [])
            label_to_text = {lab: txt for lab, txt in zip(labels, texts)}
            answer_key = (ex.get("answerKey") or "").strip()
            answer_text = label_to_text.get(answer_key, "")
            return {
                "question": (ex.get("question_stem") or "").strip(),
                "answer": answer_text.strip(),
                "fact": (ex.get("fact1") or "").strip(),
                "all_choices": texts,
            }

        ds = ds.map(_map)
        ds = ds.filter(lambda ex: bool(ex["question"]) and bool(ex["answer"]))
        return _select_max_samples(ds, max_samples)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to load OpenBookQA: %s", e)
        raise


def load_wikihow_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """Load WikiHow alternatives and reframe as Q/A style.

    Returns dataset with question, answer.
    """
    # Attempt 1: sentence-transformers/wikihow
    try:
        ds = load_dataset("sentence-transformers/wikihow", "pair", split=split)
        # fields: query (title), passage (body)
        ds = ds.map(
            lambda ex: {
                "question": (ex.get("query") or "").strip(),
                "answer": (ex.get("passage") or "").strip(),
            }
        )
        ds = ds.filter(lambda ex: bool(ex["question"]) and bool(ex["answer"]))
        return _select_max_samples(ds, max_samples)
    except Exception as e:  # noqa: BLE001
        logger.warning("Primary WikiHow alternative failed: %s", e)

    # Attempt 2: GEM/wiki_lingua English
    try:
        ds = load_dataset("GEM/wiki_lingua", "en", split=split)
        # fields: article_text, summary
        def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
            title = (ex.get("title") or "How to do it").strip()
            question = f"Explain how to {title}"
            answer = (ex.get("summary") or ex.get("article_text") or "").strip()
            return {"question": question, "answer": answer}

        ds = ds.map(_map)
        ds = ds.filter(lambda ex: bool(ex["question"]) and bool(ex["answer"]))
        return _select_max_samples(ds, max_samples)
    except Exception as e:  # noqa: BLE001
        logger.warning("Fallback WikiHow alternative failed: %s", e)
        # Return empty dataset to avoid failing pipeline
        return Dataset.from_dict({"question": [], "answer": []})


def _format_user_content(dataset_name: str, ex: Dict[str, Any]) -> str:
    if dataset_name.lower() == "eli5":
        return (
            f"Question: {ex.get('question','')}\n\n"
            f"Context: {ex.get('context','')}\n\n"
            "Provide a clear, detailed explanation suitable for someone learning this topic."
        )
    if dataset_name.lower() == "sciq":
        return (
            f"Science Question: {ex.get('question','')}\n\n"
            f"Supporting information: {ex.get('support','')}\n\n"
            "Explain the answer clearly."
        )
    if dataset_name.lower() == "openbookqa":
        return (
            f"Question: {ex.get('question','')}\n\n"
            f"Relevant fact: {ex.get('fact','')}\n\n"
            "Provide a clear explanation of the correct answer."
        )
    if dataset_name.lower() == "wikihow":
        return (
            f"Question: {ex.get('question','')}\n\n"
            "Provide a step-by-step explanation."
        )
    # default
    return f"Question: {ex.get('question','')}"


def format_for_instruction_tuning(
    dataset: Dataset, dataset_name: str, include_system_prompt: bool = True
) -> List[Dict[str, Any]]:
    """Convert a dataset to Mistral-style messages JSONL items."""
    results: List[Dict[str, Any]] = []
    if len(dataset) == 0:
        return results

    for ex in dataset:
        messages: List[Dict[str, str]] = []
        if include_system_prompt:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": _format_user_content(dataset_name, ex)})
        messages.append({"role": "assistant", "content": (ex.get("answer") or "").strip()})
        results.append({"messages": messages})

    logger.info(
        "Formatted %d examples from %s", len(results), dataset_name
    )
    return results


def save_to_jsonl(
    data: List[Dict[str, Any]], output_path: Path, metadata: Optional[Dict[str, Any]] = None
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if metadata is not None:
        meta_path = output_path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2, ensure_ascii=False)
    logger.info("Wrote %s (%d lines)", output_path, len(data))


def load_from_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")
    results: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict) or "messages" not in obj:
                    logger.warning("Skipping malformed line %d: missing messages", line_no)
                    continue
                results.append(obj)
            except json.JSONDecodeError as e:  # noqa: BLE001
                logger.warning("Skipping malformed line %d: %s", line_no, e)
                continue
    return results


def _augment_copy_with_user_append(
    examples: List[Dict[str, Any]], suffix: str, num_augmented: int
) -> List[Dict[str, Any]]:
    if not examples:
        return []
    n = min(num_augmented, len(examples))
    sampled = random.sample(examples, n)
    aug: List[Dict[str, Any]] = []
    for ex in sampled:
        msgs = ex.get("messages", [])
        new_msgs: List[Dict[str, str]] = []
        for m in msgs:
            if m.get("role") == "user":
                new_msgs.append({"role": "user", "content": m.get("content", "") + suffix})
            else:
                new_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        aug.append({"messages": new_msgs})
    return aug


def augment_with_analogies(examples: List[Dict[str, Any]], num_augmented: int = 1000) -> List[Dict[str, Any]]:
    suffix = "\n\nInclude a helpful analogy to make the concept easier to understand."
    return _augment_copy_with_user_append(examples, suffix, num_augmented)


def augment_with_examples(examples: List[Dict[str, Any]], num_augmented: int = 1000) -> List[Dict[str, Any]]:
    suffix = "\n\nProvide concrete examples to illustrate the concept."
    return _augment_copy_with_user_append(examples, suffix, num_augmented)


def augment_step_by_step(examples: List[Dict[str, Any]], num_augmented: int = 1000) -> List[Dict[str, Any]]:
    suffix = "\n\nBreak down the explanation into clear, numbered steps."
    return _augment_copy_with_user_append(examples, suffix, num_augmented)


def back_translate_augmentation(
    examples: List[Dict[str, Any]], num_augmented: int = 500, pivot_lang: str = "de"
) -> List[Dict[str, Any]]:
    try:
        from transformers import MarianMTModel, MarianTokenizer  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.warning("Transformers translation models unavailable: %s", e)
        return []

    def _load_pair(lang: str):
        src_name = f"Helsinki-NLP/opus-mt-en-{lang}"
        tgt_name = f"Helsinki-NLP/opus-mt-{lang}-en"
        tok_src = MarianTokenizer.from_pretrained(src_name)
        mod_src = MarianMTModel.from_pretrained(src_name)
        tok_tgt = MarianTokenizer.from_pretrained(tgt_name)
        mod_tgt = MarianMTModel.from_pretrained(tgt_name)
        return (tok_src, mod_src, tok_tgt, mod_tgt)

    try:
        tok_src, mod_src, tok_tgt, mod_tgt = _load_pair(pivot_lang)
    except Exception as e:  # noqa: BLE001
        logger.warning("Back-translation models failed to load: %s", e)
        return []

    if not examples:
        return []
    n = min(num_augmented, len(examples))
    sampled = random.sample(examples, n)
    device = "cpu"  # rely on CPU by default

    def _translate_batch(tokenizer, model, texts: List[str]) -> List[str]:
        if not texts:
            return []
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    augmented: List[Dict[str, Any]] = []
    batch_size = 4
    for i in range(0, len(sampled), batch_size):
        batch = sampled[i : i + batch_size]
        answers = [m["messages"][2]["content"] if len(m.get("messages", [])) >= 3 else "" for m in batch]
        try:
            fwd = _translate_batch(tok_src, mod_src, answers)
            bwd = _translate_batch(tok_tgt, mod_tgt, fwd)
        except Exception as e:  # noqa: BLE001
            logger.warning("Translation failed for a batch: %s", e)
            continue
        for ex, new_ans in zip(batch, bwd):
            msgs = ex.get("messages", [])
            new_msgs: List[Dict[str, str]] = []
            replaced = False
            for m in msgs:
                if m.get("role") == "assistant" and not replaced:
                    new_msgs.append({"role": "assistant", "content": new_ans})
                    replaced = True
                else:
                    new_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            augmented.append({"messages": new_msgs})
    return augmented


def validate_instruction_format(examples: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    valid_roles = {"system", "user", "assistant"}
    for idx, ex in enumerate(examples):
        if "messages" not in ex or not isinstance(ex["messages"], list):
            errors.append(f"Example {idx}: missing messages list")
            continue
        msgs = ex["messages"]
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        contents = [m.get("content") for m in msgs if isinstance(m, dict)]
        if any(r not in valid_roles for r in roles):
            errors.append(f"Example {idx}: invalid role present")
        if any(c is None or c == "" for c in contents):
            errors.append(f"Example {idx}: empty content present")
        if not any(m.get("role") == "user" for m in msgs):
            errors.append(f"Example {idx}: missing user message")
        if not any(m.get("role") == "assistant" for m in msgs):
            errors.append(f"Example {idx}: missing assistant message")
    return (len(errors) == 0, errors)


def _stats_lengths(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(examples)
    if total == 0:
        return {"total": 0}
    user_lens: List[int] = []
    assist_lens: List[int] = []
    for ex in examples:
        msgs = ex.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                user_lens.append(len(m.get("content", "")))
            if m.get("role") == "assistant":
                assist_lens.append(len(m.get("content", "")))
    return {
        "total": total,
        "avg_user_len": sum(user_lens) / max(1, len(user_lens)),
        "avg_assistant_len": sum(assist_lens) / max(1, len(assist_lens)),
    }


def combine_datasets(
    output_dir: Path = PROCESSED_DIR, max_samples_per_dataset: Optional[int] = None
) -> Tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets...")
    eli5_train = load_eli5_dataset("train", max_samples_per_dataset)
    sciq_train = load_sciq_dataset("train", max_samples_per_dataset)
    obqa_train = load_openbookqa_dataset("train", max_samples_per_dataset)
    wkh_train = load_wikihow_dataset("train", max_samples_per_dataset)

    logger.info("Formatting datasets...")
    eli5_formatted = format_for_instruction_tuning(eli5_train, "eli5")
    sciq_formatted = format_for_instruction_tuning(sciq_train, "sciq")
    obqa_formatted = format_for_instruction_tuning(obqa_train, "openbookqa")
    wkh_formatted = format_for_instruction_tuning(wkh_train, "wikihow")

    save_to_jsonl(eli5_formatted, output_dir / "eli5_instructions.jsonl")
    save_to_jsonl(sciq_formatted, output_dir / "sciq_instructions.jsonl")
    save_to_jsonl(obqa_formatted, output_dir / "openbookqa_instructions.jsonl")
    save_to_jsonl(wkh_formatted, output_dir / "wikihow_instructions.jsonl")

    combined = eli5_formatted + sciq_formatted + obqa_formatted + wkh_formatted
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.95)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    save_to_jsonl(train_data, output_dir / "combined_train.jsonl")
    save_to_jsonl(val_data, output_dir / "combined_val.jsonl")

    stats = {
        "eli5": _stats_lengths(eli5_formatted),
        "sciq": _stats_lengths(sciq_formatted),
        "openbookqa": _stats_lengths(obqa_formatted),
        "wikihow": _stats_lengths(wkh_formatted),
        "combined": {
            "train": {"count": len(train_data)},
            "val": {"count": len(val_data)},
        },
    }
    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return len(train_data), len(val_data), 4


def augment_datasets(
    input_dir: Path = PROCESSED_DIR,
    output_dir: Path = AUGMENTED_DIR,
    augmentation_config: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = augmentation_config or {}
    analogies_n = int(config.get("analogies", os.getenv("AUGMENTATION_SAMPLES", 1000)))
    examples_n = int(config.get("examples", os.getenv("AUGMENTATION_SAMPLES", 1000)))
    steps_n = int(config.get("step_by_step", os.getenv("AUGMENTATION_SAMPLES", 1000)))
    backtr_n = int(config.get("back_translation", 500))
    pivot = str(config.get("pivot_lang", os.getenv("BACK_TRANSLATION_PIVOT", "de")))

    train_path = input_dir / "combined_train.jsonl"
    train_data = load_from_jsonl(train_path)

    analogies = augment_with_analogies(train_data, analogies_n)
    examples = augment_with_examples(train_data, examples_n)
    step_by_step = augment_step_by_step(train_data, steps_n)
    back_translated = back_translate_augmentation(train_data, backtr_n, pivot)

    save_to_jsonl(analogies, output_dir / "with_analogies.jsonl")
    save_to_jsonl(examples, output_dir / "with_examples.jsonl")
    save_to_jsonl(step_by_step, output_dir / "step_by_step.jsonl")
    save_to_jsonl(back_translated, output_dir / "back_translated.jsonl")

    combined_augmented = train_data + analogies + examples + step_by_step + back_translated
    random.shuffle(combined_augmented)
    save_to_jsonl(combined_augmented, output_dir / "combined_augmented.jsonl")

    stats = {
        "original": len(train_data),
        "analogies": len(analogies),
        "examples": len(examples),
        "step_by_step": len(step_by_step),
        "back_translated": len(back_translated),
        "combined_augmented": len(combined_augmented),
    }
    with (output_dir / "augmentation_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data preparation for instruction tuning")
    p.add_argument("--max-samples", type=int, default=None, help="Limit per dataset")
    p.add_argument("--skip-augmentation", action="store_true")
    p.add_argument("--augment-only", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    if args.augment_only:
        stats = augment_datasets()
        print(json.dumps(stats, indent=2))
        return

    train_count, val_count, num_datasets = combine_datasets(
        PROCESSED_DIR, args.max_samples
    )
    print(
        json.dumps(
            {
                "train": train_count,
                "val": val_count,
                "datasets": num_datasets,
            },
            indent=2,
        )
    )
    if not args.skip_augmentation:
        stats = augment_datasets()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
