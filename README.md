# Reasoning Project

## Repository Structure

### Model-Specific JSONL Files
Contains the original JSONL files for each model with translations, excluding entries with empty translations.

### Sentence-Level BLEU per File
Contains sentence-level BLEU scores computed for each individual translation sample in the model-specific JSONL files. Each file includes detailed metrics:
- `score`: BLEU score
- `counts`: Counts
- `totals`: Totals
- `precisions`: Precisions
- `bp`: Brevity penalty
- `sys_len`: Prediction length
- `ref_len`: Reference length
(Metrics were gathered from: https://huggingface.co/spaces/evaluate-metric/sacrebleu)

### Mean BLEU Score per File
Contains the mean BLEU scores computed across all samples in each model-specific JSONL file, providing an overall quality assessment per model, translation method, and language pair.

## Files

### `json_sacrebleu_script.py`
Python program that computes sentence-level BLEU scores using the sacrebleu library. The script:
- Processes all JSONL files in the directory
- Supports multiple translation methods (direct_translation, teacher-CoT-translation, self-CoT-translation, teacher-Synthesized-CoT-translation)
- Uses the sacrebleu library's sentence_score method for evaluation

### `Analysis on Method Performance.pdf`
Analysis document identifying which translation method (among the three methods) achieves the best BLEU score for each language pair and model ID combination.

### `Mean BLEU Score - Sheet1.pdf`
Compiled spreadsheet containing all mean BLEU scores.

### `Number of Empty Translations - Sheet1.pdf`
Spreadsheet recording the count of empty translations per model-specific JSONL file.
