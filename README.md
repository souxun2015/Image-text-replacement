# Image Text Replacement

This project detects Chinese text in images, translates it to English and replaces the text in the image. It uses pytesseract for OCR, googletrans for translation, OpenCV for inpainting, and PIL for drawing the English text.

## Requirements

- Python 3.11+
- Tesseract OCR (installed system-wide)
- Python dependencies listed in `requirements.txt`

## Installation

```
pip install -r requirements.txt
```

## Usage

```
python text_replace.py input.jpg output.jpg
```

This will produce an output image with Chinese characters replaced by English translations.
