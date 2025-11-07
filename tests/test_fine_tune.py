import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.fine_tune import (
    create_bnb_config,
    create_lora_config,
    load_datasets,
    create_training_arguments,
)


@pytest.fixture
def mock_dataset() -> Any:
    class _MockDS:
        def __init__(self):
            self._data = [
                {"messages": [{"role": "user", "content": "hi"}]},
                {"messages": [{"role": "assistant", "content": "hello"}]},
            ]
            self.column_names = ["messages"]

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MockDS()


def test_create_bnb_config_qlora():
    cfg = create_bnb_config(use_qlora=True)
    assert cfg is not None
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True


def test_create_bnb_config_no_qlora():
    assert create_bnb_config(use_qlora=False) is None


def test_create_lora_config_defaults():
    cfg = create_lora_config()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert abs(cfg.lora_dropout - 0.05) < 1e-6
    for m in ("q_proj", "down_proj"):
        assert m in cfg.target_modules
    assert cfg.bias == "none"
    assert cfg.task_type == "CAUSAL_LM"


def test_create_lora_config_custom():
    cfg = create_lora_config(r=32, alpha=64, dropout=0.1)
    assert cfg.r == 32
    assert cfg.lora_alpha == 64
    assert abs(cfg.lora_dropout - 0.1) < 1e-6


def test_load_datasets_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_datasets(tmp_path / "missing.jsonl", tmp_path / "missing2.jsonl")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_datasets_success(tmp_path: Path, mock_dataset):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, [{"messages": []}])
    _write_jsonl(val, [{"messages": []}])

    with patch("src.fine_tune.load_dataset") as mocked:
        mocked.return_value = {"train": mock_dataset, "validation": mock_dataset}
        tr, va = load_datasets(train, val)
        assert tr is mock_dataset
        assert va is mock_dataset
        assert "messages" in tr.column_names


def test_create_training_arguments(tmp_path: Path):
    args = create_training_arguments(
        output_dir=tmp_path,
        num_epochs=3,
        batch_size=1,
        grad_accum=4,
        lr=2e-4,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
        max_seq_len=2048,
    )
    assert args.output_dir == str(tmp_path)
    assert int(args.num_train_epochs) == 3
    assert int(args.per_device_train_batch_size) == 1
    assert int(args.gradient_accumulation_steps) == 4
    assert abs(float(args.learning_rate) - 2e-4) < 1e-9
    assert args.gradient_checkpointing is True
    assert args.optim == "paged_adamw_8bit"
    assert int(args.max_seq_length) == 2048


def test_training_arguments_effective_batch_size(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    args = create_training_arguments(
        output_dir=tmp_path,
        num_epochs=1,
        batch_size=2,
        grad_accum=4,
        lr=2e-4,
        save_steps=10,
        eval_steps=10,
        logging_steps=1,
        max_seq_len=512,
    )
    # No direct field for effective batch size; this test ensures no exceptions and valid args
    assert int(args.per_device_train_batch_size) == 2
    assert int(args.gradient_accumulation_steps) == 4
