import string
import torch
import torch.nn as nn

chars = string.ascii_uppercase + string.digits
pad_token = "_"
chars = chars + pad_token
max_length = 5
num_classes = len(chars)

class CaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
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
model.load_state_dict(torch.load("captcha_model.pth", map_location=device))
model.eval()

dummy_input = torch.randn(1, 1, 25, 120).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "captcha_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    export_params=True,
    external_data=False
)