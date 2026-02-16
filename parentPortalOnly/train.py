import os
import string
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

chars = string.ascii_uppercase + string.digits
pad_token = "_"
chars = chars + pad_token
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
max_length = 5
num_classes = len(chars)

class CaptchaDataset(Dataset):
    def __init__(self, image_dir, csv_file):
        self.image_dir = image_dir
        df = pd.read_csv(csv_file, encoding="utf-8", dtype=str)
        df["text"] = df["text"].str.upper()
        df["text"] = df["text"].str.replace(r"[^A-Z0-9]", "", regex=True)
        df = df[df["text"].str.len() > 0]
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def encode_label(self, text):
        text = text.ljust(max_length, pad_token)
        return torch.tensor([char_to_idx[c] for c in text])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        img = Image.open(img_path).convert("L")
        img = img.crop((0, 0, 120, 25))
        img = self.transform(img)
        label = self.encode_label(row["text"])
        return img, label

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

dataset = CaptchaDataset("../dataset/student/captchas", "../dataset/student/labels.csv")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptchaModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(40):
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = 0
        for i in range(max_length):
            loss += criterion(outputs[:, i, :], labels[:, i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(2)
        correct += (preds == labels).all(dim=1).sum().item()
        total += labels.size(0)
    print("Epoch:", epoch, "Loss:", total_loss, "Accuracy:", correct / total)

torch.save(model.state_dict(), "captcha_model.pth")