from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.explanation_engine import create_explanation_engine, ExplanationEngine
from src.ppt_processor import process_ppt


logger = logging.getLogger(__name__)

GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() in ("true", "1", "yes")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/adapters/final")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() in ("true", "1", "yes")

_engine: Optional[ExplanationEngine] = None


def get_engine() -> ExplanationEngine:
    """Lazily create and cache the ExplanationEngine instance.

    Returns:
        ExplanationEngine: The initialized engine instance.

    Raises:
        RuntimeError: If engine initialization fails.
    """
    global _engine
    if _engine is not None:
        return _engine
    try:
        logger.info(
            "Initializing ExplanationEngine (4bit=%s) with adapters at: %s",
            LOAD_IN_4BIT,
            ADAPTER_PATH,
        )
        _engine = create_explanation_engine(
            adapter_path=ADAPTER_PATH,
            load_mistral_in_4bit=LOAD_IN_4BIT,
            load_vision_in_4bit=LOAD_IN_4BIT,
        )
        logger.info("ExplanationEngine initialized: %s", MODEL_NAME)
        return _engine
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to initialize ExplanationEngine")
        raise RuntimeError(
            "Engine initialization failed. Check adapter path, GPU memory (CUDA OOM), and environment."
        ) from e


def process_ppt_file(file_path: str, session_state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], int, str]:
    """Process uploaded PPTX file and update session state.

    Args:
        file_path: Path to uploaded PPTX file.
        session_state: Current session state dictionary.

    Returns:
        Tuple containing:
            - updated session_state dict
            - list of image paths for gallery
            - total number of slides
            - status message string
    """
    try:
        if not file_path or not Path(file_path).exists():
            return session_state, [], 0, "No file provided or file not found."
        if Path(file_path).suffix.lower() != ".pptx":
            return session_state, [], 0, "Invalid file type. Please upload a .pptx file."

        logger.info("Processing PPT: %s", file_path)
        ppt_data = process_ppt(file_path, extract_images_flag=True)
        session_state["ppt_data"] = ppt_data
        session_state["current_slide_idx"] = 0
        session_state["chat_history"] = []
        session_state["last_explanation"] = ""

        slides = ppt_data.get("slides", []) if isinstance(ppt_data, dict) else []
        gallery_images: List[str] = []
        for slide in slides:
            for img in slide.get("images", []) or []:
                path = img.get("image_path")
                if path:
                    gallery_images.append(path)

        status = f"Processed {len(slides)} slides. Found {len(gallery_images)} images."
        return session_state, gallery_images, len(slides), status
    except Exception as e:  # noqa: BLE001
        logger.exception("Error while processing PPT file")
        return session_state, [], 0, f"Error processing PPT: {e}"


def navigate_slide(direction: str, current_idx: int, total_slides: int) -> int:
    """Calculate the new slide index based on navigation direction.

    Args:
        direction: Either "prev" or "next".
        current_idx: Current slide index (0-based).
        total_slides: Total number of slides.

    Returns:
        The new slide index.
    """
    if total_slides <= 0:
        return 0
    if direction == "prev":
        return max(0, current_idx - 1)
    if direction == "next":
        return min(max(0, total_slides - 1), current_idx + 1)
    return current_idx


def get_slide_info(session_state: Dict[str, Any], slide_idx: int) -> str:
    """Return a markdown summary for the requested slide.

    Args:
        session_state: Current session state.
        slide_idx: Slide index to summarize.

    Returns:
        Markdown string with slide details.
    """
    try:
        ppt_data = session_state.get("ppt_data") if isinstance(session_state, dict) else None
        slides: List[Dict[str, Any]] = (ppt_data or {}).get("slides", [])
        if not slides or slide_idx < 0 or slide_idx >= len(slides):
            return "No slide data available"
        slide = slides[slide_idx]
        text = (slide.get("text") or "").strip()
        text_preview = (text[:500] + ("..." if len(text) > 500 else "")) if text else "(no text)"
        tables = slide.get("tables", []) or []
        images = slide.get("images", []) or []
        notes = (slide.get("notes") or "").strip()
        notes_preview = (notes[:200] + ("..." if len(notes) > 200 else "")) if notes else "(no notes)"
        return (
            f"**Slide {slide_idx + 1}**\n\n"
            f"Text: {text_preview}\n\n"
            f"Tables: {len(tables)} table(s)\n\n"
            f"Images: {len(images)} image(s)\n\n"
            f"Notes: {notes_preview}"
        )
    except Exception:  # noqa: BLE001
        return "No slide data available"


def explain_slides(
    session_state: Dict[str, Any], slide_indices: Optional[List[int]] = None
) -> Tuple[Dict[str, Any], List[Tuple[str, str]], str]:
    """Generate explanation for selected slides and update chat history.

    Args:
        session_state: Current session state dict.
        slide_indices: Optional slide indices; defaults to current slide.

    Returns:
        Tuple of (updated session_state, chat_history, explanation_text)
    """
    try:
        if not session_state or not session_state.get("ppt_data"):
            msg = "Please upload a PPT first."
            ch = session_state.get("chat_history", [])
            ch.append(("Explain", msg))
            session_state["chat_history"] = ch
            return session_state, ch, msg

        engine = get_engine()
        if slide_indices is None:
            slide_indices = [int(session_state.get("current_slide_idx", 0))]

        with gr.Progress(track_tqdm=True) as progress:  # type: ignore[arg-type]
            progress(0, desc="Preparing context...")
            result = engine.explain_concept(
                slides_data=session_state["ppt_data"],
                slide_indices=slide_indices,
                include_images=True,
            )

        explanation = result.get("explanation") if isinstance(result, dict) else None
        explanation_text = explanation if isinstance(explanation, str) else str(result)
        session_state["last_explanation"] = explanation_text
        ch = session_state.get("chat_history", [])
        ch.append((f"Explain slides {slide_indices}", explanation_text))
        session_state["chat_history"] = ch
        return session_state, ch, explanation_text
    except Exception as e:  # noqa: BLE001
        logger.exception("Error during explanation generation")
        err = f"Failed to generate explanation: {e}"
        ch = session_state.get("chat_history", [])
        ch.append(("Explain", err))
        session_state["chat_history"] = ch
        return session_state, ch, err


def generate_analogies_handler(
    session_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[str, str]], str]:
    """Generate analogies from last explanation and update chat history."""
    try:
        last = (session_state or {}).get("last_explanation") or ""
        if not last:
            msg = "No previous explanation found. Click 'Explain Concept' first."
            ch = session_state.get("chat_history", [])
            ch.append(("Generate Analogies", msg))
            session_state["chat_history"] = ch
            return session_state, ch, msg
        engine = get_engine()
        result = engine.generate_analogies(concept_text=last)
        analogies = result.get("analogies") if isinstance(result, dict) else None
        analogies_text = analogies if isinstance(analogies, str) else str(result)
        ch = session_state.get("chat_history", [])
        ch.append(("Generate Analogies", analogies_text))
        session_state["chat_history"] = ch
        return session_state, ch, analogies_text
    except Exception as e:  # noqa: BLE001
        logger.exception("Error during analogies generation")
        err = f"Failed to generate analogies: {e}"
        ch = session_state.get("chat_history", [])
        ch.append(("Generate Analogies", err))
        session_state["chat_history"] = ch
        return session_state, ch, err


def create_examples_handler(
    session_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[str, str]], str]:
    """Generate examples from last explanation and update chat history."""
    try:
        last = (session_state or {}).get("last_explanation") or ""
        if not last:
            msg = "No previous explanation found. Click 'Explain Concept' first."
            ch = session_state.get("chat_history", [])
            ch.append(("Create Examples", msg))
            session_state["chat_history"] = ch
            return session_state, ch, msg
        engine = get_engine()
        result = engine.create_examples(concept_text=last)
        examples = result.get("examples") if isinstance(result, dict) else None
        examples_text = examples if isinstance(examples, str) else str(result)
        ch = session_state.get("chat_history", [])
        ch.append(("Create Examples", examples_text))
        session_state["chat_history"] = ch
        return session_state, ch, examples_text
    except Exception as e:  # noqa: BLE001
        logger.exception("Error during examples generation")
        err = f"Failed to create examples: {e}"
        ch = session_state.get("chat_history", [])
        ch.append(("Create Examples", err))
        session_state["chat_history"] = ch
        return session_state, ch, err


def generate_quiz_handler(
    session_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[str, str]], str]:
    """Generate quiz questions from last explanation and update chat history."""
    try:
        last = (session_state or {}).get("last_explanation") or ""
        if not last:
            msg = "No previous explanation found. Click 'Explain Concept' first."
            ch = session_state.get("chat_history", [])
            ch.append(("Generate Quiz", msg))
            session_state["chat_history"] = ch
            return session_state, ch, msg
        engine = get_engine()
        result = engine.generate_questions(concept_text=last, num_questions=3)

        formatted = []
        if isinstance(result, dict) and isinstance(result.get("questions"), list):
            for i, q in enumerate(result["questions"], 1):
                question = q.get("question", "")
                options = q.get("options", []) or []
                answer = q.get("answer", "")
                rationale = q.get("rationale", "")
                options_md = "\n".join([f"- {opt}" for opt in options])
                formatted.append(
                    f"**Q{i}: {question}**\n{options_md}\n**Answer:** {answer}\n**Rationale:** {rationale}\n"
                )
            quiz_md = "\n\n".join(formatted)
        else:
            quiz_md = f"Received unexpected format for questions: {result}"

        ch = session_state.get("chat_history", [])
        ch.append(("Generate Quiz", quiz_md))
        session_state["chat_history"] = ch
        return session_state, ch, quiz_md
    except Exception as e:  # noqa: BLE001
        logger.exception("Error during quiz generation")
        err = f"Failed to generate quiz: {e}"
        ch = session_state.get("chat_history", [])
        ch.append(("Generate Quiz", err))
        session_state["chat_history"] = ch
        return session_state, ch, err


def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(title="Explainable AI Tutor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 Explainable AI Tutor\n\nUpload a PowerPoint presentation and get interactive explanations!")

        session_state = gr.State(
            value={
                "ppt_data": None,
                "current_slide_idx": 0,
                "chat_history": [],
                "last_explanation": "",
            }
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 Upload & Navigate")
                file_upload = gr.File(label="Upload PowerPoint", file_types=[".pptx"], type="filepath")
                upload_status = gr.Textbox(label="Status", interactive=False, lines=2)
                gr.Markdown("### Slide Navigation")
                slide_info = gr.Markdown("No presentation loaded")
                with gr.Row():
                    prev_btn = gr.Button("◀ Previous", size="sm")
                    slide_counter = gr.Number(label="Slide", value=1, precision=0, interactive=False)
                    next_btn = gr.Button("Next ▶", size="sm")
                slide_gallery = gr.Gallery(
                    label="Slide Images",
                    columns=1,
                    height=300,
                    object_fit="contain",
                    allow_preview=True,
                    show_label=False,
                )

            with gr.Column(scale=2):
                gr.Markdown("## 💬 Explanation History")
                chatbot = gr.Chatbot(label="Chat History", type="messages", height=600, show_copy_button=True)
                current_output = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Actions")
                gr.Markdown("Generate explanations and learning materials:")
                explain_btn = gr.Button("📖 Explain Concept", variant="primary", size="lg")
                analogy_btn = gr.Button("🔄 Generate Analogies", size="lg")
                example_btn = gr.Button("💡 Create Examples", size="lg")
                quiz_btn = gr.Button("❓ Generate Quiz", size="lg")
                gr.Markdown("---")
                gr.Markdown("### ℹ️ Instructions")
                gr.Markdown("1. Upload a .pptx file\n2. Navigate slides\n3. Click 'Explain' to start\n4. Use other buttons for more details")

        # Events
        file_upload.upload(
            fn=process_ppt_file,
            inputs=[file_upload, session_state],
            outputs=[session_state, slide_gallery, slide_counter, upload_status],
        )

        def _nav_prev(s: Dict[str, Any]) -> Dict[str, Any]:
            total = len((s.get("ppt_data") or {}).get("slides", []) or [])
            s["current_slide_idx"] = navigate_slide("prev", int(s.get("current_slide_idx", 0)), total)
            return s

        def _nav_next(s: Dict[str, Any]) -> Dict[str, Any]:
            total = len((s.get("ppt_data") or {}).get("slides", []) or [])
            s["current_slide_idx"] = navigate_slide("next", int(s.get("current_slide_idx", 0)), total)
            return s

        prev_btn.click(_nav_prev, [session_state], [session_state]).then(
            lambda s: (int(s.get("current_slide_idx", 0)) + 1, get_slide_info(s, int(s.get("current_slide_idx", 0)))),
            [session_state],
            [slide_counter, slide_info],
        )
        next_btn.click(_nav_next, [session_state], [session_state]).then(
            lambda s: (int(s.get("current_slide_idx", 0)) + 1, get_slide_info(s, int(s.get("current_slide_idx", 0)))),
            [session_state],
            [slide_counter, slide_info],
        )

        explain_btn.click(explain_slides, [session_state], [session_state, chatbot, current_output], concurrency_limit=1)
        analogy_btn.click(
            generate_analogies_handler, [session_state], [session_state, chatbot, current_output], concurrency_limit=1
        )
        example_btn.click(
            create_examples_handler, [session_state], [session_state, chatbot, current_output], concurrency_limit=1
        )
        quiz_btn.click(
            generate_quiz_handler, [session_state], [session_state, chatbot, current_output], concurrency_limit=1
        )

    return demo


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Launch Explainable AI Tutor web interface")
    parser.add_argument("--port", type=int, default=GRADIO_SERVER_PORT, help="Server port")
    parser.add_argument("--host", type=str, default=GRADIO_SERVER_NAME, help="Server host")
    parser.add_argument("--share", action="store_true", default=GRADIO_SHARE, help="Create public share link")
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH, help="Path to LoRA adapters")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    """Main entry point for launching the app."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting Explainable AI Tutor on %s:%s (share=%s)", args.host, args.port, args.share)
    logger.info("Adapters: %s | 4bit=%s", args.adapter_path, LOAD_IN_4BIT)
    demo = create_ui()
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True)


# Expose demo for hot-reload mode
try:
    demo = create_ui()
    demo.queue(default_concurrency_limit=1, max_size=20)
except Exception:
    # In module import contexts where gradio may not be available
    demo = None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        raise
