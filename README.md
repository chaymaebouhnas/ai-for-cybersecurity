# AI for Cybersecurity — PII Image Anonymizer

A Flask API that automatically finds personally identifiable information in an image — faces,
and text containing PII such as names, organizations, locations or dates — and anonymizes it by
blurring or pixelating it.

## How it works

1. **Face detection** — an OpenCV DNN face detector (a Caffe SSD model, `models/deploy.prototxt`
   + `models/res10_300x300_ssd_iter_140000.caffemodel`) locates faces in the uploaded image.
2. **Text PII detection** — [EasyOCR](https://github.com/JaidedAI/EasyOCR) extracts any text in
   the image, and [spaCy](https://spacy.io/)'s named-entity recognizer scans that text for PII
   entity types (`PERSON`, `ORG`, `GPE`, `LOC`, `DATE`, `TIME`, `MONEY`, …) — so only genuinely
   sensitive text gets flagged, not every word in the image.
3. **Anonymization** — every detected face and PII text region is either Gaussian-blurred or
   pixelated (configurable), with face-detection confidence drawn on screen for transparency
   during testing.

## API

- `GET /api/status` — reports whether the OCR, NLP and face-detection models loaded
  successfully.
- `POST /api/anonymize` — multipart form upload:
  - `image` (file, required)
  - `method` (`blur` or `pixelate`, default `blur`)
  - `confidence` (face-detection threshold, default `0.5`)

  Returns a JSON response with the anonymized image as a base64 data URL and the count of PII
  regions found.

## Running it locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
# API available at http://localhost:5001
```

The face-detection model files are already included in `models/`; on first run the app will
also verify/re-download them automatically if missing (from the OpenCV model zoo).

## Stack

Flask, Flask-CORS, OpenCV (DNN face detection), EasyOCR, spaCy (named-entity recognition), NumPy.
