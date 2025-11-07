import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ppt_processor import process_ppt
from src.explanation_engine import create_explanation_engine

SAMPLE_PPT_PATH = Path("static/samples/sample_lecture.pptx")
ADAPTER_PATH = "models/adapters/final"
LOAD_IN_4BIT = True
OUTPUT_DIR = Path("test_results")


def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("manual_test")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(OUTPUT_DIR / "manual_test.log")
    fh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def test_ppt_processing(ppt_path: str) -> Dict:
    data = process_ppt(ppt_path)
    summary = {
        "total_slides": data.get("total_slides", 0),
        "num_images": len(data.get("images", [])),
        "num_slides_records": len(data.get("slides", [])),
    }
    (OUTPUT_DIR / "ppt_processing.json").write_text(json.dumps({"summary": summary, "data_keys": list(data.keys())}, indent=2))
    return data


def test_engine_initialization():
    t0 = time.time()
    engine = create_explanation_engine(adapter_path=ADAPTER_PATH, load_mistral_in_4bit=LOAD_IN_4BIT, load_vision_in_4bit=LOAD_IN_4BIT)
    t1 = time.time()
    info = {"init_seconds": round(t1 - t0, 2)}
    (OUTPUT_DIR / "engine_init.json").write_text(json.dumps(info, indent=2))
    return engine


def test_explanation_generation(engine, slides_data) -> Dict:
    t0 = time.time()
    result = engine.explain_concept(slides_data=slides_data, slide_indices=[0, 1, 2])
    t1 = time.time()
    (OUTPUT_DIR / "explanation.txt").write_text(result.get("explanation", ""))
    (OUTPUT_DIR / "explanation.json").write_text(json.dumps({"meta": result.get("metadata", {}), "seconds": round(t1 - t0, 2)}, indent=2))
    return result


def test_analogies_generation(engine, explanation_text) -> Dict:
    t0 = time.time()
    result = engine.generate_analogies(explanation_text)
    t1 = time.time()
    (OUTPUT_DIR / "analogies.json").write_text(json.dumps({"analogies": result.get("analogies", []), "seconds": round(t1 - t0, 2)}, indent=2))
    return result


def test_examples_generation(engine, explanation_text) -> Dict:
    t0 = time.time()
    result = engine.create_examples(explanation_text)
    t1 = time.time()
    (OUTPUT_DIR / "examples.json").write_text(json.dumps({"examples": result.get("examples", []), "seconds": round(t1 - t0, 2)}, indent=2))
    return result


def test_quiz_generation(engine, explanation_text) -> Dict:
    t0 = time.time()
    result = engine.generate_questions(explanation_text, num_questions=3)
    t1 = time.time()
    (OUTPUT_DIR / "quiz.json").write_text(json.dumps(result, indent=2))
    (OUTPUT_DIR / "quiz_meta.json").write_text(json.dumps({"seconds": round(t1 - t0, 2)}, indent=2))
    return result


def test_different_explanation_styles(engine, slides_data) -> Dict:
    outputs = {}
    for style in ["comprehensive", "eli5", "step_by_step", "feynman"]:
        r = engine.explain_with_style(slides_data=slides_data, style=style)
        (OUTPUT_DIR / f"explanation_{style}.txt").write_text(r.get("explanation", ""))
        outputs[style] = {"len": len(r.get("explanation", "")), "meta": r.get("metadata", {})}
    (OUTPUT_DIR / "styles.json").write_text(json.dumps(outputs, indent=2))
    return outputs


def test_image_captioning(engine, slides_data) -> Dict:
    image_paths = []
    for s in slides_data.get("slides", []):
        for img in s.get("images", []) or []:
            p = img.get("path")
            if p:
                image_paths.append(p)
    captions = {}
    t0 = time.time()
    for p in image_paths[:5]:
        captions[p] = engine.caption_image(p)
    t1 = time.time()
    (OUTPUT_DIR / "captions.json").write_text(json.dumps({"captions": captions, "seconds": round(t1 - t0, 2)}, indent=2))
    return captions


def test_multi_slide_context(engine, slides_data) -> Dict:
    results = {}
    for n in [1, 3, 5, 10]:
        idx = list(range(min(n, len(slides_data.get("slides", [])))))
        t0 = time.time()
        r = engine.explain_concept(slides_data=slides_data, slide_indices=idx)
        t1 = time.time()
        results[str(n)] = {"len": len(r.get("explanation", "")), "seconds": round(t1 - t0, 2)}
    (OUTPUT_DIR / "multi_slide.json").write_text(json.dumps(results, indent=2))
    return results


def generate_test_report(results: Dict) -> None:
    md = ["# Manual Test Report", ""]
    for k, v in results.items():
        md.append(f"- **{k}**: {json.dumps(v)[:200]}")
    (OUTPUT_DIR / "test_report.md").write_text("\n".join(md))


def main() -> int:
    logger = setup_logging()
    if not SAMPLE_PPT_PATH.exists():
        logger.error(f"Sample PPT not found at {SAMPLE_PPT_PATH}. Please place a file there.")
        return 1

    results = {}
    try:
        slides_data = test_ppt_processing(str(SAMPLE_PPT_PATH))
        engine = test_engine_initialization()
        exp = test_explanation_generation(engine, slides_data)
        results["explanation"] = {"topic": exp.get("topic"), "len": len(exp.get("explanation", ""))}
        an = test_analogies_generation(engine, exp.get("explanation", ""))
        results["analogies"] = {"count": len(an.get("analogies", []))}
        ex = test_examples_generation(engine, exp.get("explanation", ""))
        results["examples"] = {"count": len(ex.get("examples", []))}
        qz = test_quiz_generation(engine, exp.get("explanation", ""))
        results["quiz"] = {"count": len(qz.get("questions", []))}
        styles = test_different_explanation_styles(engine, slides_data)
        results["styles"] = styles
        caps = test_image_captioning(engine, slides_data)
        results["captions"] = {"count": len(caps)}
        multi = test_multi_slide_context(engine, slides_data)
        results["multi_slide"] = multi
    except Exception as e:
        logger.exception("Manual tests encountered an error")
        results["error"] = str(e)
    finally:
        generate_test_report(results)

    logger.info("Manual tests completed. See test_results directory.")
    return 0 if "error" not in results else 1


if __name__ == "__main__":
    raise SystemExit(main())
