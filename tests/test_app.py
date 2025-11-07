import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.app import (
    get_engine,
    process_ppt_file,
    navigate_slide,
    get_slide_info,
    explain_slides,
    generate_analogies_handler,
    create_examples_handler,
    generate_quiz_handler,
)


@pytest.fixture
def mock_engine():
    engine = Mock()
    engine.explain_concept.return_value = {
        "explanation": "This is an explanation.",
        "topic": "Sample Topic",
        "metadata": {},
    }
    engine.generate_analogies.return_value = {
        "analogies": "Analogy 1; Analogy 2",
        "topic": "Sample Topic",
        "metadata": {},
    }
    engine.create_examples.return_value = {
        "examples": "Example 1; Example 2",
        "topic": "Sample Topic",
        "metadata": {},
    }
    engine.generate_questions.return_value = {
        "questions": [
            {
                "question": "What is X?",
                "options": ["A", "B", "C", "D"],
                "answer": "A",
                "rationale": "Because...",
            }
        ],
        "topic": "Sample Topic",
        "metadata": {},
    }
    return engine


@pytest.fixture
def sample_ppt_data():
    return {
        "slides": [
            {
                "text": "Slide 1 text content",
                "images": [{"image_path": "static/images/extracted/slide1_img1.png"}],
                "tables": [
                    {"rows": [["A", "B"], ["C", "D"]]},
                ],
                "notes": "Speaker notes 1",
            },
            {
                "text": "Slide 2 text content",
                "images": [
                    {"image_path": "static/images/extracted/slide2_img1.png"},
                    {"image_path": "static/images/extracted/slide2_img2.png"},
                ],
                "tables": [],
                "notes": "",
            },
        ]
    }


@pytest.fixture
def sample_session_state(sample_ppt_data):
    return {
        "ppt_data": sample_ppt_data,
        "current_slide_idx": 0,
        "chat_history": [],
        "last_explanation": "Previous explanation text",
    }


def test_get_engine_lazy_loading(monkeypatch, mock_engine):
    import src.app as app_mod

    create_called = {"count": 0}

    def fake_create(**kwargs):
        create_called["count"] += 1
        return mock_engine

    app_mod._engine = None
    monkeypatch.setattr(app_mod, "create_explanation_engine", lambda **kwargs: fake_create(**kwargs))

    e1 = app_mod.get_engine()
    e2 = app_mod.get_engine()
    assert e1 is e2
    assert create_called["count"] == 1


def test_get_engine_error_handling(monkeypatch):
    import src.app as app_mod

    app_mod._engine = None
    monkeypatch.setattr(app_mod, "create_explanation_engine", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError) as exc:
        app_mod.get_engine()
    assert "Engine initialization failed" in str(exc.value)


def test_process_ppt_file_success(monkeypatch, tmp_path, sample_ppt_data):
    import src.app as app_mod

    ppt_path = tmp_path / "test.pptx"
    ppt_path.write_bytes(b"dummy")
    monkeypatch.setattr(app_mod, "process_ppt", lambda *args, **kwargs: sample_ppt_data)

    state, images, slide_count, status = app_mod.process_ppt_file(str(ppt_path), {})

    assert isinstance(state, dict)
    assert "ppt_data" in state
    assert state["current_slide_idx"] == 0
    assert state["chat_history"] == []
    assert slide_count == 2
    assert len(images) == 3
    assert "Processed 2 slides" in status


def test_process_ppt_file_invalid_extension():
    state, images, slide_count, status = process_ppt_file("test.pdf", {})
    assert slide_count == 0
    assert "Invalid file type" in status


def test_navigate_slide_prev():
    assert navigate_slide("prev", 2, 5) == 1
    assert navigate_slide("prev", 0, 5) == 0


def test_navigate_slide_next():
    assert navigate_slide("next", 2, 5) == 3
    assert navigate_slide("next", 4, 5) == 4


def test_get_slide_info(sample_session_state):
    md = get_slide_info(sample_session_state, 0)
    assert isinstance(md, str)
    assert "Slide 1" in md
    assert "Tables:" in md
    assert "Images:" in md


def test_get_slide_info_no_data():
    md = get_slide_info({}, 0)
    assert md == "No slide data available"


def test_explain_slides(monkeypatch, mock_engine, sample_session_state):
    monkeypatch.setattr("src.app.get_engine", lambda: mock_engine)
    state, chat, text = explain_slides(sample_session_state, [0, 1])
    assert isinstance(state, dict)
    assert state["last_explanation"]
    assert chat and isinstance(chat[-1], tuple)
    mock_engine.explain_concept.assert_called_once()


def test_explain_slides_no_ppt_data(monkeypatch, mock_engine):
    state = {"chat_history": []}
    state, chat, text = explain_slides(state, None)
    assert "Please upload a PPT first" in text


def test_generate_analogies_handler(monkeypatch, mock_engine, sample_session_state):
    monkeypatch.setattr("src.app.get_engine", lambda: mock_engine)
    state, chat, text = generate_analogies_handler(sample_session_state)
    assert "Analogy" in text or isinstance(text, str)
    mock_engine.generate_analogies.assert_called_once()


def test_generate_analogies_no_explanation(monkeypatch):
    state, chat, text = generate_analogies_handler({"last_explanation": ""})
    assert "Click 'Explain Concept' first" in text


def test_create_examples_handler(monkeypatch, mock_engine, sample_session_state):
    monkeypatch.setattr("src.app.get_engine", lambda: mock_engine)
    state, chat, text = create_examples_handler(sample_session_state)
    assert "Example" in text or isinstance(text, str)
    mock_engine.create_examples.assert_called_once()


def test_generate_quiz_handler(monkeypatch, mock_engine, sample_session_state):
    monkeypatch.setattr("src.app.get_engine", lambda: mock_engine)
    state, chat, text = generate_quiz_handler(sample_session_state)
    assert "Q1" in text or "Q" in text
    assert "Answer:" in text
    assert "Rationale:" in text
    mock_engine.generate_questions.assert_called_once()


def test_generate_quiz_handler_json_error(monkeypatch, sample_session_state):
    bad_engine = Mock()
    bad_engine.generate_questions.return_value = {"error": "parse_error", "raw": "..."}
    monkeypatch.setattr("src.app.get_engine", lambda: bad_engine)
    state, chat, text = generate_quiz_handler(sample_session_state)
    assert "Received unexpected format" in text


def test_ui_creation():
    import gradio as gr
    from src.app import create_ui

    demo = create_ui()
    assert isinstance(demo, gr.Blocks)
