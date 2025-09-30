# Srmap CAPTCHA Solver

A CRNN (Convolutional Recurrent Neural Network) model trained on 5,000 CAPTCHA samples from the SRMAP Student Portal. The model automatically recognizes and solves CAPTCHAs.

---

## Endpoint

```bash
curl -X POST "http://localhost:6000/captcha" -F "file=@captcha.png"
```
- Output: Plain text CAPTCHA string
---

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- FastAPI
- Uvicorn
- Pillow
- Python-multipart

Install dependencies via pip:

```bash
pip install -r requirements.txt
```
***
| Developed By Brahmendra |
|:-------:|