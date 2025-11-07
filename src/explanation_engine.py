from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

from src.model_inference import MistralInference, load_inference_model
from src.ppt_processor import process_ppt

logger = logging.getLogger(__name__)

# Env-configurable constants
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "Salesforce/blip2-flan-t5-xl")
VISION_MODEL_LOAD_IN_4BIT = os.getenv("VISION_MODEL_LOAD_IN_4BIT", "true").lower() in ("true", "1", "yes")
MAX_SLIDES_CONTEXT = int(os.getenv("MAX_SLIDES_CONTEXT", "5"))
EXPLANATION_MAX_TOKENS = int(os.getenv("EXPLANATION_MAX_TOKENS", "512"))
ANALOGY_MAX_TOKENS = int(os.getenv("ANALOGY_MAX_TOKENS", "256"))
EXAMPLE_MAX_TOKENS = int(os.getenv("EXAMPLE_MAX_TOKENS", "384"))
QUIZ_MAX_TOKENS = int(os.getenv("QUIZ_MAX_TOKENS", "768"))
EXPLANATION_TEMPERATURE = float(os.getenv("EXPLANATION_TEMPERATURE", "0.7"))
QUIZ_TEMPERATURE = float(os.getenv("QUIZ_TEMPERATURE", "0.4"))

# System prompt (aligned with data_preparation.py intent)
SYSTEM_PROMPT = (
    "You are an expert educational tutor specializing in explaining complex concepts to "
    "undergraduate students. Provide clear, accurate, and engaging explanations using "
    "analogies, examples, and step-by-step reasoning when appropriate."
)

# Prompt templates
EXPLANATION_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Slide Content:\n{slide_context}\n\n"
    "Task: Explain this concept clearly for an undergraduate student. Follow these guidelines:\n"
    "1. Start with a simple, intuitive explanation (ELI5 style)\n"
    "2. Identify the key components and how they relate\n"
    "3. Use everyday analogies where helpful\n"
    "4. Provide concrete examples\n"
    "5. Address common misconceptions\n\n"
    "Keep your explanation clear, accurate, and engaging. Use approximately 3-5 paragraphs.\n"
)

ANALOGY_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Concept to explain:\n{concept_text}\n\n"
    "Task: Create 2-3 helpful analogies to make this concept easier to understand.\n\n"
    "For each analogy:\n"
    "- Use familiar, everyday situations\n"
    "- Explain how the analogy maps to the concept\n"
    "- Note any limitations of the analogy\n\n"
    "Format:\n"
    "**Analogy 1: [Title]**\n"
    "[Explanation]\n"
    "*Limitation: [Where the analogy breaks down]*\n\n"
    "**Analogy 2: [Title]**\n"
    "...\n"
)

EXAMPLE_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Concept:\n{concept_text}\n\n"
    "Task: Provide 2-3 concrete, real-world examples that illustrate this concept.\n\n"
    "For each example:\n"
    "- Describe a specific, realistic scenario\n"
    "- Explain how the concept applies\n"
    "- Highlight the key takeaway\n\n"
    "Format:\n"
    "**Example 1: [Title]**\n"
    "[Scenario description]\n"
    "*Key takeaway: [What this demonstrates]*\n\n"
    "**Example 2: [Title]**\n"
    "...\n"
)

STEP_BY_STEP_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Concept:\n{concept_text}\n\n"
    "Task: Break down this concept into clear, numbered steps.\n\n"
    "Provide:\n"
    "1. A step-by-step breakdown (3-7 steps)\n"
    "2. Brief explanation for each step\n"
    "3. How the steps connect to form the complete concept\n\n"
    "Format:\n"
    "**Step 1: [Title]**\n"
    "[Explanation]\n\n"
    "**Step 2: [Title]**\n"
    "...\n\n"
    "**How it all connects:**\n"
    "[Summary of the flow]\n"
)

QUIZ_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Content covered:\n{concept_text}\n\n"
    "Task: Generate 3 multiple-choice questions to assess understanding of this concept.\n\n"
    "Requirements:\n"
    "- Target undergraduate level (Bloom's: Understand and Apply)\n"
    "- 4 options per question (A, B, C, D)\n"
    "- Exactly one correct answer\n"
    "- Plausible distractors that reflect common misconceptions\n"
    "- Include brief rationale for correct answer\n\n"
    "Format (use this exact JSON structure):\n"
    "```json\n"
    "[\n"
    "  {{\n"
    "    \"question\": \"[Question text]\",\n"
    "    \"options\": [\n"
    "      \"A. [Option A]\",\n"
    "      \"B. [Option B]\",\n"
    "      \"C. [Option C]\",\n"
    "      \"D. [Option D]\"\n"
    "    ],\n"
    "    \"correct_index\": 0,\n"
    "    \"rationale\": \"[Why this answer is correct]\"\n"
    "  }},\n"
    "  ...\n"
    "]\n"
    "```\n\n"
    "Generate exactly 3 questions in valid JSON format.\n"
)


class ExplanationEngine:
    """High-level orchestrator that integrates Mistral text generation and BLIP-2 image captioning.

    This engine consumes slide data (from process_ppt) and produces educational outputs such as
    comprehensive explanations, analogies, examples, and MCQ quizzes. Image captions are generated
    on-demand with caching to reduce redundant compute.
    """

    def __init__(
        self,
        mistral_inference: Optional[MistralInference] = None,
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        load_mistral_in_4bit: bool = False,
        load_vision_in_4bit: bool = VISION_MODEL_LOAD_IN_4BIT,
        vision_model_name: str = VISION_MODEL_NAME,
    ) -> None:
        """Initialize engine with text and vision models.

        Args:
            mistral_inference: Optional pre-initialized MistralInference
            model_name: Mistral model name (if creating a new inference instance)
            adapter_path: Path to LoRA adapters
            load_mistral_in_4bit: Load Mistral in 4-bit
            load_vision_in_4bit: Load BLIP-2 in 4-bit
            vision_model_name: Hugging Face repo for BLIP-2
        """
        try:
            if mistral_inference is None:
                mistral_inference = load_inference_model(
                    model_name=model_name,
                    adapter_path=adapter_path,
                    load_in_4bit=load_mistral_in_4bit,
                )
            self.mistral_inference = mistral_inference
        except Exception as exc:
            logger.error("Failed to initialize Mistral inference: %s", exc)
            raise

        # Device selection for vision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor
        try:
            self.vision_processor = Blip2Processor.from_pretrained(vision_model_name)
        except Exception as exc:
            logger.error("Failed to load BLIP-2 processor: %s", exc)
            raise

        # BitsAndBytes for 4-bit
        bnb_config = None
        if load_vision_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Load BLIP-2 model
        try:
            if bnb_config is not None:
                self.vision_model = Blip2ForConditionalGeneration.from_pretrained(
                    vision_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                self.vision_model = Blip2ForConditionalGeneration.from_pretrained(
                    vision_model_name,
                    torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
                    device_map=("auto" if self.device == "cuda" else None),
                )
            self.vision_model.eval()
        except Exception as exc:
            logger.error("Failed to load BLIP-2 model: %s", exc)
            raise

        # Cache for image captions
        self.image_caption_cache: Dict[str, str] = {}

        logger.info(
            "ExplanationEngine initialized | vision=%s (4bit=%s) device=%s",
            vision_model_name,
            bool(bnb_config is not None),
            self.device,
        )

    # Internal helpers
    def _caption_image(self, image_path: str, use_cache: bool = True) -> str:
        """Caption an image with BLIP-2, with optional in-memory caching.

        Args:
            image_path: Path to image file
            use_cache: Whether to use and populate cache

        Returns:
            Caption string or fallback placeholder.
        """
        if use_cache and image_path in self.image_caption_cache:
            return self.image_caption_cache[image_path]
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.vision_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.vision_model.device) for k, v in inputs.items()}
            generated_ids = self.vision_model.generate(**inputs, max_new_tokens=30)
            # Per plan: decode via processor
            caption = self.vision_processor.decode(generated_ids[0], skip_special_tokens=True)
            if use_cache:
                self.image_caption_cache[image_path] = caption
            return caption
        except Exception as exc:
            logger.warning("Failed to caption image %s: %s", image_path, exc)
            return "[Image description unavailable]"

    def _build_slide_context(
        self,
        slides: List[Dict[str, Any]],
        slide_indices: Optional[List[int]] = None,
        include_images: bool = True,
    ) -> str:
        """Compose a textual context from selected slides, including tables and images.

        Respects MAX_SLIDES_CONTEXT when slide_indices is None.
        """
        selected: List[int]
        if slide_indices is None:
            max_n = min(MAX_SLIDES_CONTEXT, len(slides))
            selected = list(range(max_n))
        else:
            selected = list(slide_indices)[: MAX_SLIDES_CONTEXT]

        parts: List[str] = []
        for idx in selected:
            if idx < 0 or idx >= len(slides):
                logger.warning("Slide index out of range: %s", idx)
                continue
            slide = slides[idx]
            parts.append(f"--- Slide {slide.get('slide_number', idx + 1)} ---")
            text = (slide.get("text_content") or "").strip()
            if text:
                parts.append(f"Text: {text}")
            tables = slide.get("tables") or []
            if tables:
                for t in tables:
                    # Simple markdown-like table formatting
                    if isinstance(t, list) and t:
                        header = " | ".join(str(c) for c in t[0])
                        parts.append(f"Tables: | {header} |")
                        for row in t[1:]:
                            parts.append("| " + " | ".join(str(c) for c in row) + " |")
            notes = slide.get("notes")
            if notes:
                parts.append(f"Speaker Notes: {notes}")
            if include_images:
                for im in slide.get("images", []) or []:
                    img_path = im.get("image_path")
                    if not img_path:
                        continue
                    caption = self._caption_image(img_path)
                    parts.append(f"Image: {caption}")
            parts.append("")
        return "\n".join(parts).strip()

    def _extract_topic(
        self,
        slides: List[Dict[str, Any]],
        slide_indices: Optional[List[int]] = None,
    ) -> str:
        """Heuristic topic extraction from first selected slide text."""
        if not slides:
            return "Concept from presentation"
        indices = (slide_indices or [0])
        first_idx = 0 if not indices else max(0, min(indices[0], len(slides) - 1))
        text = (slides[first_idx].get("text_content") or "").strip()
        if not text:
            return "Concept from presentation"
        if len(text) <= 50:
            return text
        # Take first sentence-ish or first 100 chars
        stop = min(len(text), 100)
        dot = text.find(".")
        if 0 < dot < 100:
            stop = dot + 1
        return text[:stop]

    # Public APIs
    def explain_concept(
        self,
        ppt_path: Optional[str] = None,
        slides_data: Optional[Dict[str, Any]] = None,
        slide_indices: Optional[List[int]] = None,
        include_images: bool = True,
        temperature: float = EXPLANATION_TEMPERATURE,
        max_tokens: int = EXPLANATION_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Generate a comprehensive explanation from PPT content.

        Returns a structured dict: {explanation, topic, slides_used, metadata}
        """
        try:
            if slides_data is None:
                if not ppt_path or not Path(ppt_path).exists():
                    raise FileNotFoundError("ppt_path must be a valid file when slides_data is None")
                slides_data = process_ppt(ppt_path, extract_images_flag=include_images)

            slides = slides_data.get("slides", [])
            if not slides:
                raise ValueError("No slides found in slides_data")

            slide_context = self._build_slide_context(slides, slide_indices, include_images)
            topic = self._extract_topic(slides, slide_indices)

            user_content = EXPLANATION_TEMPLATE.format(topic=topic, slide_context=slide_context)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            explanation = self.mistral_inference.generate(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            used_indices = slide_indices if slide_indices is not None else list(range(min(len(slides), MAX_SLIDES_CONTEXT)))
            return {
                "explanation": explanation,
                "topic": topic,
                "slides_used": used_indices,
                "metadata": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "include_images": include_images,
                },
            }
        except Exception as exc:
            logger.error("explain_concept failed: %s", exc)
            return {
                "error": str(exc),
                "explanation": "",
                "topic": "",
                "slides_used": slide_indices or [],
                "metadata": {"temperature": temperature, "max_tokens": max_tokens, "include_images": include_images},
            }

    def generate_analogies(
        self,
        concept_text: str,
        topic: Optional[str] = None,
        temperature: float = EXPLANATION_TEMPERATURE,
        max_tokens: int = ANALOGY_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Generate 2-3 analogies for a concept."""
        try:
            topic_val = topic or (concept_text.split(".")[0].strip() or "Concept")
            user_content = ANALOGY_TEMPLATE.format(topic=topic_val, concept_text=concept_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            analogies = self.mistral_inference.generate(
                messages, max_new_tokens=max_tokens, temperature=temperature, do_sample=True
            )
            return {"analogies": analogies, "topic": topic_val, "metadata": {"temperature": temperature, "max_tokens": max_tokens}}
        except Exception as exc:
            logger.error("generate_analogies failed: %s", exc)
            return {"error": str(exc), "analogies": "", "topic": topic or "", "metadata": {}}

    def create_examples(
        self,
        concept_text: str,
        topic: Optional[str] = None,
        temperature: float = EXPLANATION_TEMPERATURE,
        max_tokens: int = EXAMPLE_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Generate concrete real-world examples for a concept."""
        try:
            topic_val = topic or (concept_text.split(".")[0].strip() or "Concept")
            user_content = EXAMPLE_TEMPLATE.format(topic=topic_val, concept_text=concept_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            examples = self.mistral_inference.generate(
                messages, max_new_tokens=max_tokens, temperature=temperature, do_sample=True
            )
            return {"examples": examples, "topic": topic_val, "metadata": {"temperature": temperature, "max_tokens": max_tokens}}
        except Exception as exc:
            logger.error("create_examples failed: %s", exc)
            return {"error": str(exc), "examples": "", "topic": topic or "", "metadata": {}}

    def generate_questions(
        self,
        concept_text: str,
        topic: Optional[str] = None,
        num_questions: int = 3,
        temperature: float = QUIZ_TEMPERATURE,
        max_tokens: int = QUIZ_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Generate MCQ questions in JSON and parse/validate the result."""
        try:
            topic_val = topic or (concept_text.split(".")[0].strip() or "Concept")
            tmpl = QUIZ_TEMPLATE
            user_content = tmpl.format(topic=topic_val, concept_text=concept_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            raw = self.mistral_inference.generate(
                messages, max_new_tokens=max_tokens, temperature=temperature, do_sample=True
            )

            parsed: Any = None
            try:
                parsed = json.loads(raw)
            except Exception:
                # Try extracting JSON from code block
                try:
                    start = raw.find("```json")
                    if start != -1:
                        start = raw.find("[", start)
                        end = raw.rfind("]")
                        if start != -1 and end != -1 and end > start:
                            snippet = raw[start : end + 1]
                            parsed = json.loads(snippet)
                except Exception:
                    parsed = None

            if not isinstance(parsed, list):
                raise ValueError("Failed to parse questions JSON")

            required = {"question", "options", "correct_index", "rationale"}
            for item in parsed:
                if not isinstance(item, dict) or not required.issubset(item.keys()):
                    raise ValueError("Invalid question structure")

            if num_questions and len(parsed) != num_questions:
                logger.warning("Model returned %d questions, expected %d", len(parsed), num_questions)

            return {"questions": parsed, "topic": topic_val, "metadata": {"num_questions": num_questions, "temperature": temperature}}
        except Exception as exc:
            logger.error("generate_questions failed: %s", exc)
            return {"error": str(exc), "raw": locals().get("raw", ""), "topic": topic or "", "metadata": {}}

    def explain_with_style(
        self,
        ppt_path: Optional[str] = None,
        slides_data: Optional[Dict[str, Any]] = None,
        slide_indices: Optional[List[int]] = None,
        style: str = "comprehensive",
        include_images: bool = True,
    ) -> Dict[str, Any]:
        """Convenience wrapper to generate explanations with different styles."""
        style_key = (style or "").lower()
        if style_key == "step_by_step":
            # Use step by step template as the concept_text, using context as concept_text
            data = slides_data
            if data is None and ppt_path:
                data = process_ppt(ppt_path, extract_images_flag=include_images)
            if not data or not data.get("slides"):
                return {"error": "No slides found", "metadata": {"style": style_key}}
            slides = data["slides"]
            concept_text = self._build_slide_context(slides, slide_indices, include_images)
            topic = self._extract_topic(slides, slide_indices)
            user_content = STEP_BY_STEP_TEMPLATE.format(topic=topic, concept_text=concept_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            text = self.mistral_inference.generate(messages, max_new_tokens=EXPLANATION_MAX_TOKENS, temperature=EXPLANATION_TEMPERATURE, do_sample=True)
            return {"explanation": text, "topic": topic, "slides_used": slide_indices or list(range(min(len(slides), MAX_SLIDES_CONTEXT))), "metadata": {"style": style_key}}

        # Comprehensive as default, with optional modifiers
        result = self.explain_concept(
            ppt_path=ppt_path,
            slides_data=slides_data,
            slide_indices=slide_indices,
            include_images=include_images,
            temperature=EXPLANATION_TEMPERATURE,
            max_tokens=EXPLANATION_MAX_TOKENS,
        )
        if "error" in result:
            return result
        extra_note = ""
        if style_key == "eli5":
            extra_note = "\n\nPlease explain like I'm 5 years old. Use very simple language and everyday analogies."
        elif style_key == "feynman":
            extra_note = "\n\nUse the Feynman technique: explain simply, identify gaps, refine the explanation."

        if extra_note:
            # Re-run with modified user content
            slides = (slides_data or {}).get("slides") if slides_data else None
            if not slides and ppt_path:
                processed = process_ppt(ppt_path, extract_images_flag=include_images)
                slides = processed.get("slides", [])
            if not slides:
                return result
            topic = result.get("topic") or self._extract_topic(slides, slide_indices)
            slide_context = self._build_slide_context(slides, slide_indices, include_images)
            user_content = EXPLANATION_TEMPLATE.format(topic=topic, slide_context=slide_context) + extra_note
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            text = self.mistral_inference.generate(messages, max_new_tokens=EXPLANATION_MAX_TOKENS, temperature=EXPLANATION_TEMPERATURE, do_sample=True)
            result["explanation"] = text
        # annotate style
        result.setdefault("metadata", {})["style"] = style_key if style_key in {"comprehensive", "eli5", "step_by_step", "feynman"} else "comprehensive"
        return result

    def clear_image_cache(self) -> None:
        """Clear the in-memory image caption cache."""
        self.image_caption_cache.clear()


def create_explanation_engine(
    model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    load_mistral_in_4bit: bool = False,
    load_vision_in_4bit: bool = VISION_MODEL_LOAD_IN_4BIT,
    vision_model_name: str = VISION_MODEL_NAME,
) -> ExplanationEngine:
    """Factory to instantiate ExplanationEngine with common defaults."""
    return ExplanationEngine(
        model_name=model_name,
        adapter_path=adapter_path,
        load_mistral_in_4bit=load_mistral_in_4bit,
        load_vision_in_4bit=load_vision_in_4bit,
        vision_model_name=vision_model_name,
    )
