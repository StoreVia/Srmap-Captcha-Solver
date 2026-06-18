from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import string
import io
import uvicorn

chars = string.ascii_uppercase + string.digits
pad_token = "_"
chars = chars + pad_token
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
max_length = 5
num_classes = len(chars)

class CaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 10))
        )
        self.classifier = nn.Linear(128 * 10, max_length * num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(-1, max_length, num_classes)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CaptchaModel().to(device)
model.load_state_dict(torch.load("srm_captcha.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

def predict_captcha(image):
    image = image.convert("L")
    image = image.crop((0, 0, 120, 25))
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        preds = output.argmax(2)[0]

    text = "".join(idx_to_char[idx.item()] for idx in preds)
    return text.replace("_", "")

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    text = predict_captcha(image)
    return {"prediction": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)