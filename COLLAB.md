# Run Explainable AI Tutor on Google Colab

This guide shows how to run the project in Google Colab using a GPU runtime. It covers dependency setup, environment variables, running notebooks, and launching the Gradio app with a public link.

## Prerequisites
- Colab runtime: Runtime → Change runtime type → Hardware accelerator → GPU
- Prefer Colab Pro (A100/L4) for VRAM. T4 (16GB) may work in 4-bit with a smaller vision model.

## Quick Start (copy each block into a Colab cell)

### 1) Clone your repository (or upload the folder)
If your code is on GitHub, replace the URL below. Otherwise, upload files manually to Colab and skip this.

```bash
# Example (replace with your repo URL)
# git clone https://github.com/<user>/<repo>.git
# cd <repo>
```

If you uploaded files manually, ensure the current working directory contains:
- `src/`, `notebooks/`, `requirements.txt`, `pyproject.toml`

### 2) Install dependencies (Colab-friendly)
Colab often ships with a specific CUDA/Torch build. The safest path is to use the matching PyTorch wheel for CUDA 12.1 and pin transformers/peft versions required by this project.

```bash
# Optional: Reinstall PyTorch for CUDA 12.1 (fits most Colab GPUs)
pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu121

# Core libs pinned for PEFT compatibility
pip install -q transformers==4.49.0 peft==0.17.1 accelerate bitsandbytes sentencepiece protobuf

# ML + evaluation
pip install -q datasets==4.3.0 trl==0.15.2 evaluate sacrebleu rouge-score

# Vision + PPT + UI
pip install -q python-pptx==1.0.0 Pillow==11.2.1 gradio==5.49.1

# Utilities
pip install -q tqdm pandas matplotlib seaborn jupyter ipykernel
```

Notes:
- If a later Colab image conflicts, restart runtime and re-run the cells.
- If you see CUDA errors, try skipping the PyTorch reinstall and use the preinstalled torch.

### 3) Set environment variables for Colab
Use 4-bit loading and a smaller BLIP-2 variant to reduce VRAM. You can switch back to `Salesforce/blip2-flan-t5-xl` if your GPU has enough memory.

```python
%env MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
%env ADAPTER_PATH=./models/adapters/final
%env LOAD_IN_4BIT=true

# Vision model (smaller, more Colab-friendly). Switch to flan-t5-xl if you have >=24GB VRAM
%env VISION_MODEL_NAME=Salesforce/blip2-opt-2.7b
%env VISION_MODEL_LOAD_IN_4BIT=true

# App/engine runtime knobs
%env MAX_SLIDES_CONTEXT=3
%env EXPLANATION_MAX_TOKENS=512
%env ANALOGY_MAX_TOKENS=256
%env EXAMPLE_MAX_TOKENS=384
%env QUIZ_MAX_TOKENS=640
%env EXPLANATION_TEMPERATURE=0.7
%env QUIZ_TEMPERATURE=0.4

# Gradio (Colab needs a public link)
%env GRADIO_SERVER_PORT=7860
%env GRADIO_SERVER_NAME=0.0.0.0
%env GRADIO_SHARE=true
```

### 4) (Optional) Mount Google Drive for data/models
```python
from google.colab import drive
drive.mount('/content/drive')

# Example: point adapters to a Drive folder to avoid re-downloads
# %env ADAPTER_PATH=/content/drive/MyDrive/padhAi/models/adapters/final
```

### 5) Run notebooks
You can open and run these directly in Colab:
- `notebooks/explore_datasets.ipynb`
- `notebooks/fine_tuning_experiments.ipynb` (requires GPU with enough VRAM)
- `notebooks/test_explanation_engine.ipynb` (uses both text + vision models)
- `notebooks/test_gradio_ui.ipynb` (lightweight sanity checks for the UI)

If a notebook relies on local paths, adjust them to the Colab working directory.

### 6) Launch the Gradio app (public share link)
Run from the project root in Colab. This will print a public `gradio.live` URL in the output cell.

```bash
python -m src.app --host 0.0.0.0 --port 7860 --share
```

Upload a `.pptx` through the UI, navigate slides, then click:
- Explain Concept
- Generate Analogies
- Create Examples
- Generate Quiz

## Tips for Colab VRAM
- Prefer `%env VISION_MODEL_NAME=Salesforce/blip2-opt-2.7b` on small GPUs.
- Keep `%env LOAD_IN_4BIT=true` and `%env VISION_MODEL_LOAD_IN_4BIT=true`.
- Reduce `%env MAX_SLIDES_CONTEXT=2` if you hit OOM.
- Close other notebooks using the GPU.

## Troubleshooting
- Out of memory (CUDA): lower `MAX_SLIDES_CONTEXT`, ensure 4-bit, use the smaller BLIP-2 model.
- Import/ABI issues: restart runtime after installs, then re-run setup cells.
- No Gradio link: ensure `--share` is set or `%env GRADIO_SHARE=true`.
- Large model downloads: first run may take several minutes to fetch models from Hugging Face.

## Minimal one-cell bootstrap (advanced)
If you only need the UI quickly (assumes project files already present in cwd):
```bash
pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu121 \
  && pip install -q transformers==4.49.0 peft==0.17.1 accelerate bitsandbytes sentencepiece protobuf \
  datasets==4.3.0 trl==0.15.2 evaluate sacrebleu rouge-score python-pptx==1.0.0 Pillow==11.2.1 gradio==5.49.1 \
  tqdm pandas matplotlib seaborn \
  && python - <<'PY'
import os
os.environ.update({
  'MODEL_NAME': 'mistralai/Mistral-7B-Instruct-v0.1',
  'ADAPTER_PATH': './models/adapters/final',
  'LOAD_IN_4BIT': 'true',
  'VISION_MODEL_NAME': 'Salesforce/blip2-opt-2.7b',
  'VISION_MODEL_LOAD_IN_4BIT': 'true',
  'GRADIO_SERVER_PORT': '7860',
  'GRADIO_SERVER_NAME': '0.0.0.0',
})
import subprocess; subprocess.run(['python','-m','src.app','--host','0.0.0.0','--port','7860','--share'], check=True)
PY
```
