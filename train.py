import os
import torch
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

TARGET_W = 120
TARGET_H = 25

def crop_captcha(img):
    w, h = img.size
    if w > TARGET_W or h > TARGET_H:
        img = img.crop((0, 0, TARGET_W, TARGET_H))
    return img

class CaptchaDataset(Dataset):
    def __init__(self, roots):
        self.samples = []
        for root in roots:
            df = pd.read_csv(os.path.join(root, "labels.csv"), header=None)
            img_dir = os.path.join(root, "captchas")
            for img, label in df.values:
                label = str(label).strip().upper()
                if label and all(c in CHARS for c in label):
                    path = os.path.join(img_dir, img)
                    if os.path.exists(path):
                        self.samples.append((path, label))
        self.tf = T.Compose([
            T.Grayscale(),
            T.Resize((32, 120)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("L")
        img = crop_captcha(img)
        img = self.tf(img)
        tgt = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)
        return img, tgt, len(tgt)

def collate(batch):
    imgs, labels, lengths = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.tensor(lengths, dtype=torch.long)

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
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(self.cnn(x))
        b, c, _, w = x.shape
        x = x.view(b, c, w).permute(2, 0, 1)
        x, _ = self.rnn(x)
        return self.fc(x)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CaptchaDataset(["dataset/parent", "dataset/parent"])
    dl = DataLoader(
        ds,
        batch_size=128,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    model = CRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(12):
        model.train()
        total = 0.0
        for imgs, labels, lengths in dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(imgs)
                logp = out.log_softmax(2)
                in_len = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long, device=device)
                loss = loss_fn(logp, labels, in_len, lengths)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        print(f"Epoch {epoch+1} Loss {total:.4f}")

    torch.save(model.state_dict(), "captcha_crnn.pth")