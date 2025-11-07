import json
from pathlib import Path
import random
import tempfile

import pytest
from datasets import Dataset

from src.data_preparation import (
    augment_datasets,
    augment_step_by_step,
    augment_with_analogies,
    augment_with_examples,
    back_translate_augmentation,
    combine_datasets,
    format_for_instruction_tuning,
    load_from_jsonl,
    load_eli5_dataset,
    load_openbookqa_dataset,
    load_sciq_dataset,
    load_wikihow_dataset,
    save_to_jsonl,
    validate_instruction_format,
)


@pytest.fixture
def sample_eli5_data() -> Dataset:
    data = {
        "title": [
            "Why is the sky blue?",
            "How do airplanes fly?",
            "What is photosynthesis?",
            "Why do we dream?",
            "How does Wi-Fi work?",
        ],
        "selftext": ["", "", "", "", ""],
        "answers": [
            {"text": ["Because of Rayleigh scattering", "Blue paint"], "score": [10, 1]},
            {"text": ["Lift from wings"], "score": [5]},
            {"text": ["Plants convert light to energy"], "score": [7]},
            {"text": ["Unclear brain processing"], "score": [2]},
            {"text": ["Radio waves and routers"], "score": [3]},
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_sciq_data() -> Dataset:
    data = {
        "question": ["What is H2O?", "What planet is known as the Red Planet?"],
        "correct_answer": ["Water", "Mars"],
        "support": ["H2O is the chemical formula for water.", "Mars appears red due to iron oxide."],
        "distractor1": ["Oxygen", "Jupiter"],
        "distractor2": ["Hydrogen", "Venus"],
        "distractor3": ["Carbon", "Mercury"],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_instruction_examples():
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a tutor."},
                {"role": "user", "content": "Explain gravity."},
                {"role": "assistant", "content": "Gravity is a force of attraction."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a tutor."},
                {"role": "user", "content": "Explain photosynthesis."},
                {"role": "assistant", "content": "Plants convert light into chemical energy."},
            ]
        },
    ]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_load_eli5_dataset_structure(monkeypatch, sample_eli5_data):
    def fake_load_dataset(name, *args, **kwargs):
        assert name == "eli5" or name == "sentence-transformers/eli5" or name == "rexarski/eli5_category"
        # Return the primary mock for canonical path
        if name == "eli5":
            return sample_eli5_data
        raise RuntimeError("secondary not needed")

    monkeypatch.setattr("src.data_preparation.load_dataset", fake_load_dataset)
    ds = load_eli5_dataset("train", max_samples=3)
    assert isinstance(ds, Dataset)
    assert set(["question", "context", "answer"]).issubset(set(ds.column_names))
    assert len(ds) == 3
    for ex in ds:
        assert ex["question"] and ex["answer"]


def test_load_sciq_dataset_structure(monkeypatch, sample_sciq_data):
    def fake_load_dataset(name, *args, **kwargs):
        assert name == "allenai/sciq"
        return sample_sciq_data

    monkeypatch.setattr("src.data_preparation.load_dataset", fake_load_dataset)
    ds = load_sciq_dataset("train")
    assert isinstance(ds, Dataset)
    assert set(["question", "answer", "support", "distractors"]).issubset(set(ds.column_names))
    for ex in ds:
        assert ex["question"] and ex["answer"]


def test_load_openbookqa_dataset_structure(monkeypatch):
    data = {
        "question_stem": ["What do plants need?"],
        "choices": [{"label": ["A", "B", "C", "D"], "text": ["Water", "Sand", "Plastic", "Steel"]}],
        "answerKey": ["A"],
        "fact1": ["Plants need water to survive."],
    }
    fake = Dataset.from_dict(data)

    def fake_load_dataset(name, config, split):
        assert name == "allenai/openbookqa" and config == "main"
        return fake

    monkeypatch.setattr("src.data_preparation.load_dataset", fake_load_dataset)
    ds = load_openbookqa_dataset("train")
    assert isinstance(ds, Dataset)
    assert set(["question", "answer", "fact", "all_choices"]).issubset(set(ds.column_names))
    for ex in ds:
        assert ex["answer"] == "Water"


def test_format_for_instruction_tuning_basic(sample_eli5_data):
    # convert sample_eli5_data to minimal mapped structure expected by formatter
    ds = sample_eli5_data.map(lambda ex: {
        "question": ex["title"],
        "context": ex.get("selftext", ""),
        "answer": ex["answers"]["text"][0],
    })
    formatted = format_for_instruction_tuning(ds, "eli5")
    assert isinstance(formatted, list) and formatted
    for item in formatted:
        assert "messages" in item
        roles = [m["role"] for m in item["messages"]]
        assert roles[0] == "system" and roles[1] == "user" and roles[2] == "assistant"


def test_format_for_instruction_tuning_no_system_prompt(sample_eli5_data):
    ds = sample_eli5_data.map(lambda ex: {
        "question": ex["title"],
        "context": ex.get("selftext", ""),
        "answer": ex["answers"]["text"][0],
    })
    formatted = format_for_instruction_tuning(ds, "eli5", include_system_prompt=False)
    for item in formatted:
        roles = [m["role"] for m in item["messages"]]
        assert roles[0] == "user" and roles[1] == "assistant"


def test_save_and_load_jsonl(sample_instruction_examples, tmp_path: Path):
    out = tmp_path / "test.jsonl"
    save_to_jsonl(sample_instruction_examples, out)
    assert out.exists()
    loaded = load_from_jsonl(out)
    assert loaded == sample_instruction_examples


def test_load_from_jsonl_malformed(tmp_path: Path, sample_instruction_examples):
    bad = tmp_path / "bad.jsonl"
    with bad.open("w", encoding="utf-8") as f:
        f.write(json.dumps(sample_instruction_examples[0]) + "\n")
        f.write("{not json}\n")
        f.write(json.dumps({"foo": 1}) + "\n")
        f.write(json.dumps(sample_instruction_examples[1]) + "\n")
    loaded = load_from_jsonl(bad)
    assert len(loaded) == 2
    for item in loaded:
        assert "messages" in item


def test_augment_with_analogies(sample_instruction_examples):
    aug = augment_with_analogies(sample_instruction_examples, num_augmented=2)
    assert len(aug) == 2
    for item in aug:
        user_msgs = [m for m in item["messages"] if m["role"] == "user"]
        assert any("analogy" in m["content"].lower() for m in user_msgs)


def test_augment_with_examples(sample_instruction_examples):
    aug = augment_with_examples(sample_instruction_examples, num_augmented=2)
    assert len(aug) == 2
    for item in aug:
        user_msgs = [m for m in item["messages"] if m["role"] == "user"]
        assert any("examples" in m["content"].lower() for m in user_msgs)


def test_augment_step_by_step(sample_instruction_examples):
    aug = augment_step_by_step(sample_instruction_examples, num_augmented=2)
    assert len(aug) == 2
    for item in aug:
        user_msgs = [m for m in item["messages"] if m["role"] == "user"]
        assert any("steps" in m["content"].lower() for m in user_msgs)


def test_back_translate_augmentation(monkeypatch, sample_instruction_examples):
    class FakeTok:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, texts, return_tensors=None, padding=True, truncation=True):
            return {"input_ids": [[0]] * len(texts)}

        def batch_decode(self, outputs, skip_special_tokens=True):
            # just return texts of same length with suffix
            return ["PARA" for _ in outputs]

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def generate(self, **kwargs):
            # produce list same size as inputs
            n = len(kwargs.get("input_ids", [[0]]))
            return [[0]] * n

    monkeypatch.setattr("src.data_preparation.MarianTokenizer", FakeTok)
    monkeypatch.setattr("src.data_preparation.MarianMTModel", FakeModel)

    aug = back_translate_augmentation(sample_instruction_examples, num_augmented=2)
    assert len(aug) <= 2
    if aug:
        # assistant message is modified
        for a in aug:
            msgs = a["messages"]
            assert any(m["role"] == "assistant" for m in msgs)


def test_validate_instruction_format_valid(sample_instruction_examples):
    ok, errs = validate_instruction_format(sample_instruction_examples)
    assert ok and errs == []


def test_validate_instruction_format_invalid():
    bad = [
        {},
        {"messages": [{"role": "user"}]},
        {"messages": [{"role": "badrole", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}]},
    ]
    ok, errs = validate_instruction_format(bad)
    assert not ok
    assert errs


def test_combine_datasets(monkeypatch, tmp_path: Path):
    # mock loaders to avoid network
    eli5 = Dataset.from_dict({"question": ["Q1"], "context": ["C1"], "answer": ["A1"]})
    sciq = Dataset.from_dict({"question": ["Q2"], "answer": ["A2"], "support": ["S2"], "distractors": [["d1","d2","d3"]]})
    obqa = Dataset.from_dict({"question": ["Q3"], "answer": ["A3"], "fact": ["F3"], "all_choices": [["A","B","C","D"]]})
    wkh = Dataset.from_dict({"question": ["Q4"], "answer": ["A4"]})

    monkeypatch.setattr("src.data_preparation.load_eli5_dataset", lambda *a, **k: eli5)
    monkeypatch.setattr("src.data_preparation.load_sciq_dataset", lambda *a, **k: sciq)
    monkeypatch.setattr("src.data_preparation.load_openbookqa_dataset", lambda *a, **k: obqa)
    monkeypatch.setattr("src.data_preparation.load_wikihow_dataset", lambda *a, **k: wkh)

    train_count, val_count, num = combine_datasets(output_dir=tmp_path, max_samples_per_dataset=10)

    assert num == 4
    assert (tmp_path / "eli5_instructions.jsonl").exists()
    assert (tmp_path / "sciq_instructions.jsonl").exists()
    assert (tmp_path / "openbookqa_instructions.jsonl").exists()
    assert (tmp_path / "wikihow_instructions.jsonl").exists()
    assert (tmp_path / "combined_train.jsonl").exists()
    assert (tmp_path / "combined_val.jsonl").exists()
    assert (tmp_path / "dataset_stats.json").exists()


def test_augment_datasets(monkeypatch, tmp_path: Path):
    # create minimal processed data
    base = [{"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}]
    processed = tmp_path
    (processed).mkdir(parents=True, exist_ok=True)
    save_to_jsonl(base, processed / "combined_train.jsonl")

    class FakeTok:
        @classmethod
        def from_pretrained(cls, name):
            return cls()
        def __call__(self, texts, return_tensors=None, padding=True, truncation=True):
            return {"input_ids": [[0]] * len(texts)}
        def batch_decode(self, outputs, skip_special_tokens=True):
            return ["BT" for _ in outputs]
    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()
        def generate(self, **kwargs):
            n = len(kwargs.get("input_ids", [[0]]))
            return [[0]] * n

    monkeypatch.setattr("src.data_preparation.MarianTokenizer", FakeTok)
    monkeypatch.setattr("src.data_preparation.MarianMTModel", FakeModel)

    stats = augment_datasets(input_dir=processed, output_dir=processed / "aug")
    assert (processed / "aug" / "with_analogies.jsonl").exists()
    assert (processed / "aug" / "with_examples.jsonl").exists()
    assert (processed / "aug" / "step_by_step.jsonl").exists()
    assert (processed / "aug" / "back_translated.jsonl").exists()
    assert (processed / "aug" / "combined_augmented.jsonl").exists()
    assert isinstance(stats, dict)


def test_max_samples_limit(monkeypatch):
    data = {
        "title": [f"Q{i}" for i in range(100)],
        "selftext": [""] * 100,
        "answers": [{"text": ["A"], "score": [1]}] * 100,
    }
    fake = Dataset.from_dict(data)

    def fake_load_dataset(name, *args, **kwargs):
        return fake

    monkeypatch.setattr("src.data_preparation.load_dataset", fake_load_dataset)
    ds = load_eli5_dataset(max_samples=10)
    assert len(ds) == 10


def test_empty_dataset_handling():
    empty = Dataset.from_dict({"question": [], "answer": []})
    out = format_for_instruction_tuning(empty, "wikihow")
    assert out == []


def test_full_pipeline_integration(monkeypatch, tmp_path: Path):
    # Mock all external datasets
    eli5 = Dataset.from_dict({"question": ["Q1"], "context": ["C1"], "answer": ["A1"]})
    sciq = Dataset.from_dict({"question": ["Q2"], "answer": ["A2"], "support": ["S2"], "distractors": [["d1","d2","d3"]]})
    obqa = Dataset.from_dict({"question": ["Q3"], "answer": ["A3"], "fact": ["F3"], "all_choices": [["A","B","C","D"]]})
    wkh = Dataset.from_dict({"question": ["Q4"], "answer": ["A4"]})

    monkeypatch.setattr("src.data_preparation.load_eli5_dataset", lambda *a, **k: eli5)
    monkeypatch.setattr("src.data_preparation.load_sciq_dataset", lambda *a, **k: sciq)
    monkeypatch.setattr("src.data_preparation.load_openbookqa_dataset", lambda *a, **k: obqa)
    monkeypatch.setattr("src.data_preparation.load_wikihow_dataset", lambda *a, **k: wkh)

    train_count, val_count, _ = combine_datasets(output_dir=tmp_path, max_samples_per_dataset=5)
    assert train_count + val_count == 4

    # Mock translation
    class FakeTok:
        @classmethod
        def from_pretrained(cls, name):
            return cls()
        def __call__(self, texts, return_tensors=None, padding=True, truncation=True):
            return {"input_ids": [[0]] * len(texts)}
        def batch_decode(self, outputs, skip_special_tokens=True):
            return ["BT" for _ in outputs]
    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()
        def generate(self, **kwargs):
            n = len(kwargs.get("input_ids", [[0]]))
            return [[0]] * n

    monkeypatch.setattr("src.data_preparation.MarianTokenizer", FakeTok)
    monkeypatch.setattr("src.data_preparation.MarianMTModel", FakeModel)

    stats = augment_datasets(input_dir=tmp_path, output_dir=tmp_path / "aug")
    assert stats["combined_augmented"] >= stats["original"]
