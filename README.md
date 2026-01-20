# SRMAP CAPTCHA Solver

This project is designed to solve **both Student Portal and Parent Portal CAPTCHAs** used in the SRMAP system using a **single optimized model and API**.

A high-accuracy CAPTCHA solver built using a **CRNN (Convolutional Recurrent Neural Network)** and deployed with **FastAPI + ONNX Runtime**.

---

## Overview

SRMAP uses two visually similar CAPTCHA formats:

| Portal | Original Size |
|------|---------------|
| Student Portal | ~220 × 30 |
| Parent Portal | ~120 × 25 |

This solver handles **both types automatically** using a unified preprocessing pipeline.

---

## How It Works

1. A CAPTCHA image is uploaded via the API.
2. If the image is larger than **120 × 25** (Student Portal):
   - The image is **cropped from the top-left** to `120 × 25`.
3. If the image is already **120 × 25** (Parent Portal):
   - No cropping is applied.
4. The image is resized to **32 × 120** and normalized.
5. The processed image is passed to a **CRNN model** exported as a **single-file ONNX model**.
6. The model predicts characters using **CTC decoding**.
7. The solved CAPTCHA text is returned as **plain text**.

This approach ensures:
- One model for both portals
- Minimal preprocessing work
- Consistent accuracy
- Fast inference

---

## Model Architecture

The CAPTCHA solver uses a **CRNN architecture**, which consists of:

- **CNN backbone** for feature extraction
- **Bidirectional LSTM** layers for sequence modeling
- **CTC (Connectionist Temporal Classification)** for alignment-free decoding

### Key Characteristics
- Trained on ~7,500 labeled CAPTCHA images
- Fixed-width inference (32 × 120)
- Robust to spacing differences between portals
- Exported to ONNX for lightweight deployment

---

## API Endpoint

### Solve CAPTCHA

```bash
curl -X POST "http://localhost:6000/captcha" -F "file=@captcha.png"
```

### Response
- **Type:** `text/plain`
- **Output:** Solved CAPTCHA string

Example:
```
A7F2K
```

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- Pillow
- python-multipart
- onnxruntime
- torchvision
- numpy

---

## Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## Running the API

```bash
python api.py
```

The server will start on:

```
http://localhost:6000
```

---

## Notes

- The ONNX model is **self-contained** (single file).
- No PyTorch dependency is required for inference.
- Designed for production deployment (VPS / Docker / serverless).
- Optimized for speed and accuracy.

---

## Disclaimer

This project is intended for **educational and automation purposes only**.  
Ensure compliance with the SRMAP portal’s terms of service before use.

---

| Developed By **Brahmendra** |
|:---------------------------:|
