from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
DEFAULT_ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/adapters/final")
DEFAULT_MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


class MistralInference:
    """Wrapper for loading a (fine-tuned) Mistral model and generating responses."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.adapter_path = adapter_path or DEFAULT_ADAPTER_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_adapters = False

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            )

        # Load tokenizer first
        tok_source = self.adapter_path if (adapter_path and Path(self.adapter_path).exists()) else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=(torch.bfloat16 if bnb_config else "auto"),
        )

        # Load adapters if present
        if adapter_path and Path(self.adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.use_adapters = True

        # Ensure eval mode
        self.model.eval()
        logger.info(
            "Loaded model for inference: base=%s adapters=%s device=%s",
            self.model_name,
            self.use_adapters,
            self.device,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        if stop_strings:
            for s in stop_strings:
                idx = response.find(s)
                if idx != -1:
                    response = response[:idx]
                    break
        return response.strip()

    def generate_from_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs,
        )

    def batch_generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int = 256,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        results: List[str] = []
        for out_ids, in_ids in zip(outputs, inputs["input_ids"]):
            gen_ids = out_ids[len(in_ids) :]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(text.strip())
        return results

    def get_model_info(self) -> Dict[str, Any]:
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            total_params = -1
        try:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        except Exception:
            trainable_params = -1
        return {
            "model_name": self.model_name,
            "adapter_path": self.adapter_path if self.use_adapters else None,
            "use_adapters": self.use_adapters,
            "device": str(self.model.device),
            "num_parameters": int(total_params) if total_params != -1 else total_params,
            "trainable_parameters": int(trainable_params) if trainable_params != -1 else trainable_params,
        }

    def merge_and_save(self, output_path: str) -> None:
        if not self.use_adapters:
            raise RuntimeError("Cannot merge: no adapters loaded.")
        merged = self.model.merge_and_unload()
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(out, safe_serialization=True)
        self.tokenizer.save_pretrained(out)
        logger.info("Merged model saved to: %s", out)


def load_inference_model(
    model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
    device: Optional[str] = None,
) -> MistralInference:
    return MistralInference(
        model_name=model_name,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        device=device,
    )
