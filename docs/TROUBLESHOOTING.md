# Troubleshooting Guide

## Table of Contents
1. Installation Issues
2. Model Loading Issues
3. Memory Issues (OOM)
4. Performance Issues
5. PPT Processing Issues
6. Explanation Quality Issues
7. Gradio UI Issues
8. Network and Connectivity Issues
9. Data Preparation Issues
10. Fine-tuning Issues

## 1. Installation Issues
- pip conflicts: use exact versions from requirements.txt
- PyTorch GPU not detected: check CUDA and drivers

## 2. Model Loading Issues
- Adapters missing: ensure `models/adapters/final/` exists
- HF token missing: set `HF_TOKEN`

## 3. Memory Issues (OOM)
- Enable 4-bit for text and vision models
- Reduce `MAX_SLIDES_CONTEXT`
- Lower `max_new_tokens`

## 4. Performance Issues
- First request is slow due to lazy model loading
- Ensure GPU is used; avoid CPU inference

## 5. PPT Processing Issues
- Only `.pptx` supported
- Ensure `static/images/extracted` writable

## 6. Explanation Quality Issues
- Increase max tokens for longer outputs
- Lower temperature for quiz stability

## 7. Gradio UI Issues
- Port conflicts, missing adapters, import errors

## 8. Network and Connectivity Issues
- Bind to `0.0.0.0` for remote access

## 9. Data Preparation Issues
- Retry dataset downloads; use cache

## 10. Fine-tuning Issues
- Reduce batch size, sequence length; enable QLoRA

## Debugging Tips
- Run with debug: `python -m src.app --log-level DEBUG`
- Monitor GPU: `watch -n 1 nvidia-smi`
- Check disk: `df -h`
