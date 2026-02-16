import os
import time
import string
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict

chars = string.ascii_uppercase + string.digits + "_"
idx_to_char = {i: c for i, c in enumerate(chars)}

so = ort.SessionOptions()
so.intra_op_num_threads = os.cpu_count()
so.inter_op_num_threads = os.cpu_count()
so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession("captcha_model.onnx", sess_options=so)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

image_dir = "../dataset/student/captchas"
csv_path = "../dataset/student/labels.csv"

df = pd.read_csv(csv_path, dtype=str)
df["text"] = df["text"].str.upper()
df["text"] = df["text"].str.replace(r"[^A-Z0-9]", "", regex=True)
df = df[df["text"].str.len() > 0]
df = df.reset_index(drop=True)

total = len(df)
correct = 0
char_correct = 0
char_total = 0
wrong = []
confusion = defaultdict(int)

start_time = time.time()

for _, row in df.iterrows():
    img_path = os.path.join(image_dir, row["filename"])
    img = Image.open(img_path).convert("L")
    img = img.crop((0, 0, 120, 25))
    img = transform(img).unsqueeze(0).numpy()

    outputs = session.run(None, {"input": img})
    pred = np.argmax(outputs[0], axis=2)[0]
    pred_text = "".join([idx_to_char[i] for i in pred]).replace("_", "")
    gt_text = row["text"]

    if pred_text == gt_text:
        correct += 1
    else:
        wrong.append((gt_text, pred_text))

    for g, p in zip(gt_text.ljust(5, "_"), pred_text.ljust(5, "_")):
        if g == p:
            char_correct += 1
        else:
            confusion[(g, p)] += 1
        char_total += 1

end_time = time.time()

elapsed = end_time - start_time

print("Total:", total)
print("Exact correct:", correct)
print("Exact accuracy:", correct / total)
print("Character accuracy:", char_correct / char_total)
print("Time seconds:", elapsed)
print("Captchas per minute:", (total / elapsed) * 60)

print("\nFirst 30 wrong predictions:")
for gt, pred in wrong[:30]:
    print("GT:", gt, "Pred:", pred)

print("\nTop 20 confusions:")
sorted_conf = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
for (g, p), count in sorted_conf[:20]:
    print("GT:", g, "Pred:", p, "Count:", count)
