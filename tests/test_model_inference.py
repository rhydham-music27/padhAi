from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.model_inference import MistralInference, load_inference_model


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "</s>"
    tok.apply_chat_template.side_effect = lambda msgs, tokenize=False, add_generation_prompt=True: (
        "USER: " + (msgs[0]["content"] if isinstance(msgs, list) and msgs and "content" in msgs[0] else "")
    )
    def _encode(x, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    tok.side_effect = None
    tok.__call__.side_effect = _encode
    tok.decode.side_effect = lambda ids, skip_special_tokens=True: "decoded-text"
    tok.batch_decode.side_effect = lambda ids, skip_special_tokens=True: ["decoded-text"] * len(ids)
    tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_model():
    model = MagicMock()
    def _generate(**kwargs):
        return torch.tensor([[1, 2, 3, 4, 5, 6]])
    model.generate.side_effect = _generate
    model.eval.return_value = None
    model.device = torch.device("cpu")
    return model


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_mistral_inference_init_with_adapters(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer, tmp_path):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model
    MockPeft.from_pretrained.return_value = mock_model

    adapters = tmp_path / "adapters"
    adapters.mkdir(parents=True, exist_ok=True)

    inf = MistralInference(model_name="base", adapter_path=str(adapters), load_in_4bit=False, device="cpu")
    assert inf.use_adapters is True
    assert inf.model is mock_model
    assert inf.tokenizer is mock_tokenizer


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_mistral_inference_init_without_adapters(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer, tmp_path):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    assert inf.use_adapters is False


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_generate_with_messages(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    out = inf.generate(messages=[{"role": "user", "content": "test"}], max_new_tokens=5)
    assert isinstance(out, str)
    mock_tokenizer.apply_chat_template.assert_called()
    mock_model.generate.assert_called()


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_generate_with_temperature(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    inf.generate(messages=[{"role": "user", "content": "x"}], temperature=0.9, do_sample=True)
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs.get("temperature") == 0.9
    assert kwargs.get("do_sample") is True


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_generate_greedy(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    inf.generate(messages=[{"role": "user", "content": "x"}], do_sample=False)
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs.get("do_sample") is False


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_generate_from_text(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    out = inf.generate_from_text("What is AI?")
    assert isinstance(out, str)


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_batch_generate(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    outs = inf.batch_generate([
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ])
    assert isinstance(outs, list) and len(outs) == 2


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_get_model_info(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    info = inf.get_model_info()
    assert set(["model_name", "device", "use_adapters"]).issubset(info.keys())


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_merge_and_save_with_adapters(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer, tmp_path):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    # Simulate adapters present by creating path and making PeftModel return same mock
    adapters = tmp_path / "adapters"
    adapters.mkdir(parents=True, exist_ok=True)

    merged_mock = MagicMock()
    def _merge():
        return merged_mock
    mock_model.merge_and_unload.side_effect = _merge

    with patch("src.model_inference.PeftModel.from_pretrained", return_value=mock_model):
        inf = MistralInference(model_name="base", adapter_path=str(adapters), load_in_4bit=False, device="cpu")
        out_dir = tmp_path / "merged"
        inf.merge_and_save(str(out_dir))
        merged_mock.save_pretrained.assert_called()


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_merge_and_save_without_adapters(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer, tmp_path):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    with pytest.raises(RuntimeError):
        inf.merge_and_save(str(tmp_path / "merged"))


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_load_inference_model_factory(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    inf = load_inference_model(model_name="base")
    assert isinstance(inf, MistralInference)


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_device_selection(monkeypatch, MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    inf = MistralInference(model_name="base", adapter_path=None)
    assert inf.device in ("cuda", str(mock_model.device))


@patch("src.model_inference.AutoTokenizer")
@patch("src.model_inference.AutoModelForCausalLM")
@patch("src.model_inference.PeftModel")
def test_generate_with_stop_strings(MockPeft, MockModel, MockTok, mock_model, mock_tokenizer):
    MockTok.from_pretrained.return_value = mock_tokenizer
    MockModel.from_pretrained.return_value = mock_model
    mock_tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: "hello END world"

    inf = MistralInference(model_name="base", adapter_path=None, load_in_4bit=False, device="cpu")
    out = inf.generate(messages=[{"role": "user", "content": "x"}], stop_strings=["END", "STOP"])
    assert out == "hello"
