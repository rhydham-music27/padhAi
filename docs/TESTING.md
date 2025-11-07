# Testing Guide

## Table of Contents
1. Testing Philosophy
2. Test Structure
3. Unit Tests
4. Integration Tests
5. End-to-End Tests
6. Manual Testing
7. Performance Testing
8. Quality Assurance
9. Continuous Integration
10. Test Coverage

## 1. Testing Philosophy
- Unit, integration, end-to-end, and manual layers
- Mock expensive ops
- Test error handling and edge cases

## 2. Test Structure
```
tests/
  test_integration_e2e.py
scripts/
  test_complete_workflow.py
  benchmark_performance.py
  validate_models.py
```

## 3. Unit Tests
Run:
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## 4. Integration Tests
```bash
pytest tests/test_integration_e2e.py -v -s --log-cli-level=INFO
```

## 5. End-to-End Tests
Manual:
```bash
cp your_presentation.pptx static/samples/sample_lecture.pptx
python scripts/test_complete_workflow.py
```

## 6. Manual Testing
Checklist includes PPT upload, slide navigation, explanations, features, styles, errors, performance, and UI.

## 7. Performance Testing
```bash
python scripts/benchmark_performance.py
```

## 8. Quality Assurance
```bash
python scripts/validate_models.py
```

## 9. Continuous Integration
Example GitHub Actions with pytest + coverage.

## 10. Test Coverage
- Target >80% overall, >90% for core modules.
- Use `--cov-report=term-missing` to find gaps.
