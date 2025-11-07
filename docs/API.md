# Explainable AI Tutor - API Documentation

## Table of Contents

1. PPT Processor API
2. Explanation Engine API
3. Model Inference API
4. Gradio App API
5. Data Preparation API
6. Fine-tuning API

---

## 1. PPT Processor API

Module: `src.ppt_processor`

Function: `process_ppt(pptx_path, extract_images_flag=True, output_dir='static/images/extracted')`
- Purpose: Process PowerPoint file and extract text, images, tables, and notes
- Parameters:
  - `pptx_path` (str): Path to .pptx file
  - `extract_images_flag` (bool): Whether to extract images
  - `output_dir` (str): Directory for extracted images
- Returns: Dict with keys `file_path`, `total_slides`, `slides`, `images`, `processing_errors`
- Example:
```python
from src.ppt_processor import process_ppt
result = process_ppt("presentation.pptx")
print(f"Processed {result['total_slides']} slides")
```
- Error Handling: Returns error dict if processing fails

---

## 2. Explanation Engine API

Module: `src.explanation_engine`

Class: `ExplanationEngine`

Method: `explain_concept(ppt_path=None, slides_data=None, slide_indices=None, include_images=True, temperature=0.7, max_tokens=512)`
- Purpose: Generate comprehensive explanation from PPT content
- Returns: Dict with `explanation`, `topic`, `slides_used`, `metadata`
- Example:
```python
from src.explanation_engine import create_explanation_engine
engine = create_explanation_engine(adapter_path="models/adapters/final", load_mistral_in_4bit=True)
result = engine.explain_concept(ppt_path="lecture.pptx", slide_indices=[0,1,2], temperature=0.7)
print(result["explanation"])
```

Method: `generate_analogies(concept_text, topic=None, temperature=0.7, max_tokens=256)`
- Purpose: Generate 2-3 analogies for a concept

Method: `create_examples(concept_text, topic=None, temperature=0.7, max_tokens=384)`
- Purpose: Generate concrete real-world examples

Method: `generate_questions(concept_text, topic=None, num_questions=3, temperature=0.4, max_tokens=768)`
- Purpose: Generate multiple-choice quiz questions

Method: `explain_with_style(ppt_path=None, slides_data=None, slide_indices=None, style='comprehensive', include_images=True)`
- Purpose: Generate explanation with specific style

Function: `create_explanation_engine(model_name=None, adapter_path=None, load_mistral_in_4bit=False, load_vision_in_4bit=True, vision_model_name='Salesforce/blip2-flan-t5-xl')`
- Purpose: Factory to create ExplanationEngine

---

## 3. Model Inference API

Module: `src.model_inference`

Class: `MistralInference`

Method: `generate(messages, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True, stop_strings=None, **kwargs)`
- Purpose: Generate response from messages array

---

## 4. Gradio App API

Module: `src.app`

Function: `create_ui()`
- Returns: `gr.Blocks`

Function: `main()`
- Usage: `python -m src.app --port 7860 --host 0.0.0.0`

---

## 5. Data Preparation API

Module: `src.data_preparation`

Function: `combine_datasets(output_dir='data/processed', max_samples_per_dataset=None)`

Function: `augment_datasets(input_dir='data/processed', output_dir='data/augmented', augmentation_config=None)`

---

## 6. Fine-tuning API

Module: `src.fine_tune`

Function: `main()`
- Usage: `python -m src.fine_tune --num-epochs 3 --learning-rate 2e-4`

---

### Best Practices
1. Use 4-bit quantization for memory efficiency
2. Lower temperature for quiz generation (0.4)
3. Cache image captions to avoid re-processing
4. Process 3-5 slides at a time for best context

### Common Patterns

Pattern: Complete workflow
```python
from src.ppt_processor import process_ppt
from src.explanation_engine import create_explanation_engine
slides_data = process_ppt("lecture.pptx")
engine = create_explanation_engine(adapter_path="models/adapters/final", load_mistral_in_4bit=True)
res = engine.explain_concept(slides_data=slides_data)
```

Error Handling:
```python
res = engine.explain_concept(ppt_path="file.pptx")
if "error" in res:
    print("Error:", res["error"]) 
```
