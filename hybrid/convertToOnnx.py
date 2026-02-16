import torch
from torch import nn

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARS) + 1

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2,1))
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2, keepdim=True)
        b, c, _, w = x.shape
        x = x.view(b, c, w).permute(2, 0, 1)
        x, _ = self.rnn(x)
        return self.fc(x)

model = CRNN()
model.load_state_dict(torch.load("captcha_crnn.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 1, 32, 120)

torch.onnx.export(
    model,
    dummy,
    "captcha_crnn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {1: "batch"}
    },
    opset_version=17
)

print("ONNX export successful.")
