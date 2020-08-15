# ArmCor
ArmCor is a Python library for detecting and correcting OCR errors for Armenian texts.
## Installation

```bash
pip install armcor
```

## Usage

```python
from armcor import ocr
ocr.detect_errors(tokens) # returns list of labels, 
                             # 1 if word contains OCR errors, and 0 otherwise
ocr.correct_errors(tokens, labels) # returns list of corrected words
