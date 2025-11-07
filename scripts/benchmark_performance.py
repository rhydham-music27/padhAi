import sys, json, time, statistics
from pathlib import Path
from typing import Dict, List
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    class _TorchShim:
        def cuda_is_available(self):
            return False
    torch = _TorchShim()  # type: ignore

from src.ppt_processor import process_ppt
from src.explanation_engine import create_explanation_engine

BENCHMARK_PPTS = [
    "static/samples/small.pptx",
    "static/samples/medium.pptx",
    "static/samples/large.pptx",
]
ITERATIONS = 3
OUT_DIR = Path("benchmark_results")


def measure_memory_usage() -> Dict:
    proc = psutil.Process()
    mem = proc.memory_info().rss
    gpu = 0
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            gpu = int(torch.cuda.memory_allocated(0))
    except Exception:
        gpu = 0
    return {"rss_bytes": mem, "gpu_bytes": gpu}


def benchmark_ppt_processing(ppt_path: str) -> Dict:
    times = []
    for _ in range(ITERATIONS):
        t0 = time.time(); _ = process_ppt(ppt_path); t1 = time.time()
        times.append(t1 - t0)
    return {"avg_seconds": round(statistics.mean(times), 3)}


def benchmark_engine_initialization() -> Dict:
    t0 = time.time(); mem0 = measure_memory_usage()
    engine = create_explanation_engine(load_mistral_in_4bit=True, load_vision_in_4bit=True)
    mem1 = measure_memory_usage(); t1 = time.time()
    return {"seconds": round(t1 - t0, 2), "mem_before": mem0, "mem_after": mem1, "ok": bool(engine)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    results["engine_init"] = benchmark_engine_initialization()
    for ppt in BENCHMARK_PPTS:
        if Path(ppt).exists():
            results[f"ppt_{Path(ppt).stem}"] = benchmark_ppt_processing(ppt)
    (OUT_DIR / "report.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "report.md").write_text("\n".join(["# Benchmark Report", *[f"- **{k}**: {v}" for k, v in results.items()]]))


if __name__ == "__main__":
    main()
