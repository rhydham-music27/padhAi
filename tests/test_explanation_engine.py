import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.explanation_engine import ExplanationEngine, create_explanation_engine


@pytest.fixture
def mock_mistral_inference():
    mock = Mock()
    mock.generate = Mock(return_value="Sample explanation text")
    mock.get_model_info = Mock(return_value={"model_name": "mistral", "use_adapters": True})
    return mock


@pytest.fixture
def mock_vision_processor(monkeypatch):
    proc = Mock()
    # __call__ returns tensor-like dict
    proc.return_value = {"pixel_values": MagicMock()}
    proc.decode = Mock(return_value="A sample image caption")

    def _from_pretrained(name):
        return proc

    with patch("src.explanation_engine.Blip2Processor.from_pretrained", side_effect=_from_pretrained):
        yield proc


@pytest.fixture
def mock_vision_model(monkeypatch):
    model = Mock()
    model.generate = Mock(return_value=[[1, 2, 3]])
    model.eval = Mock(return_value=None)
    model.device = "cpu"

    def _from_pretrained(name, **kwargs):
        return model

    with patch("src.explanation_engine.Blip2ForConditionalGeneration.from_pretrained", side_effect=_from_pretrained):
        yield model


@pytest.fixture
def sample_slides_data(tmp_path):
    slides = [
        {
            "slide_number": 1,
            "text_content": "Introduction to Photosynthesis. Plants convert light to energy.",
            "tables": [[["Factor", "Role"], ["Light", "Energy source"]]],
            "notes": "Focus on chlorophyll.",
            "images": [],
        },
        {
            "slide_number": 2,
            "text_content": "Chloroplast structure and function.",
            "tables": [],
            "notes": None,
            "images": [],
        },
        {
            "slide_number": 3,
            "text_content": "Calvin cycle overview.",
            "tables": [],
            "notes": None,
            "images": [],
        },
    ]
    return {
        "file_path": "test.pptx",
        "total_slides": 3,
        "slides": slides,
        "images": [],
        "processing_errors": [],
    }


def _init_engine(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    return ExplanationEngine(mistral_inference=mock_mistral_inference)


def test_explanation_engine_init(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    assert engine.mistral_inference is mock_mistral_inference
    assert engine.vision_processor is mock_vision_processor
    assert engine.vision_model is mock_vision_model
    assert isinstance(engine.image_caption_cache, dict)


def test_explanation_engine_init_creates_mistral(mock_vision_processor, mock_vision_model):
    with patch("src.explanation_engine.load_inference_model") as loader:
        mi = Mock()
        loader.return_value = mi
        engine = ExplanationEngine(model_name="test", adapter_path="test/path")
        assert engine.mistral_inference is mi
        loader.assert_called_with(model_name="test", adapter_path="test/path", load_in_4bit=False)


def test_caption_image(tmp_path, mock_mistral_inference, mock_vision_processor, mock_vision_model, monkeypatch):
    # Create dummy image
    img_path = tmp_path / "img.png"
    try:
        from PIL import Image

        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)
    except Exception:
        pytest.skip("Pillow image creation failed in this environment")

    # Mock processor call output tensor and decode
    def _proc_call(images=None, return_tensors=None):
        return {"pixel_values": MagicMock()}

    mock_vision_processor.__call__ = Mock(side_effect=_proc_call)

    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    caption1 = engine._caption_image(str(img_path))
    assert isinstance(caption1, str) and len(caption1) > 0
    # Cached on second call
    caption2 = engine._caption_image(str(img_path))
    assert caption2 == caption1
    # generate called only once
    assert mock_vision_model.generate.call_count == 1


def test_caption_image_error_handling(mock_mistral_inference, mock_vision_processor, mock_vision_model, monkeypatch):
    with patch("PIL.Image.open", side_effect=Exception("boom")):
        engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
        cap = engine._caption_image("nonexistent.jpg")
        assert cap == "[Image description unavailable]"


def test_build_slide_context(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    ctx = engine._build_slide_context(sample_slides_data["slides"], slide_indices=[0, 1], include_images=False)
    assert isinstance(ctx, str)
    assert "--- Slide 1 ---" in ctx
    assert "--- Slide 2 ---" in ctx
    assert "Introduction to Photosynthesis" in ctx


def test_build_slide_context_with_images(mock_mistral_inference, mock_vision_processor, mock_vision_model, tmp_path):
    # Build slides with images
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    from PIL import Image

    Image.new("RGB", (4, 4)).save(img1)
    Image.new("RGB", (4, 4)).save(img2)

    slides = [
        {"slide_number": 1, "text_content": "Slide with images", "tables": [], "notes": None, "images": [
            {"image_path": str(img1)}
        ]},
        {"slide_number": 2, "text_content": "Slide with more images", "tables": [], "notes": None, "images": [
            {"image_path": str(img2)}
        ]},
    ]

    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    with patch.object(engine, "_caption_image", side_effect=["cap1", "cap2"]) as cap:
        ctx = engine._build_slide_context(slides, include_images=True)
        assert "Image: cap1" in ctx and "Image: cap2" in ctx
        assert cap.call_count == 2


def test_build_slide_context_respects_max_slides(mock_mistral_inference, mock_vision_processor, mock_vision_model, monkeypatch):
    slides = [
        {"slide_number": i + 1, "text_content": f"S{i+1}", "tables": [], "notes": None, "images": []}
        for i in range(10)
    ]
    # Temporarily set MAX_SLIDES_CONTEXT to 3 within module
    with patch("src.explanation_engine.MAX_SLIDES_CONTEXT", 3):
        engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
        ctx = engine._build_slide_context(slides, slide_indices=None)
        assert ctx.count("--- Slide") == 3


def test_extract_topic(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    topic = engine._extract_topic(sample_slides_data["slides"])  # type: ignore[arg-type]
    assert isinstance(topic, str)
    assert len(topic) > 0


def test_explain_concept_with_slides_data(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.explain_concept(slides_data=sample_slides_data, slide_indices=[0, 1], include_images=False)
    assert set(["explanation", "topic", "slides_used", "metadata"]).issubset(out.keys())
    assert isinstance(out["explanation"], str)
    assert out["slides_used"] == [0, 1]
    mock_mistral_inference.generate.assert_called()


def test_explain_concept_with_ppt_path(tmp_path, mock_mistral_inference, mock_vision_processor, mock_vision_model):
    pptx_file = tmp_path / "file.pptx"
    pptx_file.write_bytes(b"fake")
    with patch("src.explanation_engine.process_ppt", return_value={
        "file_path": str(pptx_file),
        "total_slides": 1,
        "slides": [{"slide_number": 1, "text_content": "A", "tables": [], "notes": None, "images": []}],
        "images": [],
        "processing_errors": [],
    }) as proc:
        engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
        out = engine.explain_concept(ppt_path=str(pptx_file), include_images=False)
        proc.assert_called()
        assert isinstance(out, dict)


def test_explain_concept_temperature_and_max_tokens(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    engine.explain_concept(slides_data=sample_slides_data, temperature=0.9, max_tokens=256, include_images=False)
    args, kwargs = mock_mistral_inference.generate.call_args
    assert kwargs.get("temperature") == 0.9
    assert kwargs.get("max_new_tokens") == 256


def test_generate_analogies(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.generate_analogies(concept_text="Photosynthesis is...", topic="Photosynthesis")
    assert set(["analogies", "topic", "metadata"]).issubset(out.keys())
    mock_mistral_inference.generate.assert_called()


def test_create_examples(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.create_examples(concept_text="Machine learning is...", topic="ML")
    assert set(["examples", "topic", "metadata"]).issubset(out.keys())
    mock_mistral_inference.generate.assert_called()


def test_generate_questions(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    mock_mistral_inference.generate.return_value = json.dumps([
        {"question": "What is X?", "options": ["A", "B", "C", "D"], "correct_index": 0, "rationale": "Because..."}
    ])
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.generate_questions(concept_text="Concept...", num_questions=1)
    assert "questions" in out and isinstance(out["questions"], list)
    q = out["questions"][0]
    for k in ["question", "options", "correct_index", "rationale"]:
        assert k in q


def test_generate_questions_json_parsing_error(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    mock_mistral_inference.generate.return_value = "not json"
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.generate_questions(concept_text="Concept...", num_questions=3)
    assert "error" in out
    assert "raw" in out


def test_generate_questions_extracts_from_markdown(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    payload = """Here you go:\n```json\n[{\n  \"question\": \"Q?\", \n  \"options\": [\"A\",\"B\",\"C\",\"D\"], \n  \"correct_index\": 1, \n  \"rationale\": \"R\"\n}]\n```"""
    mock_mistral_inference.generate.return_value = payload
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.generate_questions(concept_text="Concept...", num_questions=1)
    assert "questions" in out and len(out["questions"]) == 1


def test_explain_with_style_comprehensive(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.explain_with_style(slides_data=sample_slides_data, style="comprehensive", include_images=False)
    assert "metadata" in out and out["metadata"].get("style") == "comprehensive"


def test_explain_with_style_eli5(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    with patch.object(engine, "explain_concept", wraps=engine.explain_concept) as expl:
        out = engine.explain_with_style(slides_data=sample_slides_data, style="eli5", include_images=False)
        assert out.get("metadata", {}).get("style") == "eli5"
        assert mock_mistral_inference.generate.called


def test_explain_with_style_step_by_step(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.explain_with_style(slides_data=sample_slides_data, style="step_by_step", include_images=False)
    assert "explanation" in out
    assert out.get("metadata", {}).get("style") == "step_by_step"


def test_explain_with_style_invalid(mock_mistral_inference, mock_vision_processor, mock_vision_model, sample_slides_data):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    out = engine.explain_with_style(slides_data=sample_slides_data, style="invalid_style", include_images=False)
    assert out.get("metadata", {}).get("style") == "comprehensive"


def test_create_explanation_engine_factory(mock_vision_processor, mock_vision_model):
    with patch("src.explanation_engine.ExplanationEngine", wraps=ExplanationEngine) as cls:
        eng = create_explanation_engine(model_name="test")
        assert isinstance(eng, ExplanationEngine)


def test_image_caption_cache_clearing(mock_mistral_inference, mock_vision_processor, mock_vision_model):
    engine = ExplanationEngine(mistral_inference=mock_mistral_inference)
    engine.image_caption_cache["a.jpg"] = "cap"
    assert engine.image_caption_cache
    engine.clear_image_cache()
    assert not engine.image_caption_cache
