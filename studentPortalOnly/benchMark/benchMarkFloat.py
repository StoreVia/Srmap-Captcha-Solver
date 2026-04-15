import os
import time
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

chars = string.ascii_uppercase + string.digits + "_"
idx_to_char = {i: c for i, c in enumerate(chars)}

IMAGE_DIR = "../../dataset/student/captchas"
CSV_PATH = "../../dataset/student/labels.csv"
MODEL_PATH = "../models/captcha_model_float16.tflite"

df = pd.read_csv(CSV_PATH, dtype=str)
df["text"] = df["text"].str.upper()
df["text"] = df["text"].str.replace(r"[^A-Z0-9]", "", regex=True)
df = df[df["text"].str.len() > 0]
df = df.reset_index(drop=True)

def preprocess(img_path):
    img = Image.open(img_path).convert("L")
    img = img.crop((0, 0, 120, 25))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.reshape(1, 25, 120, 1)
    return img

def decode(output):
    pred = np.argmax(output, axis=2)[0]
    return "".join([idx_to_char[i] for i in pred]).replace("_", "")

def worker(row):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_path = os.path.join(IMAGE_DIR, row["filename"])
    img = preprocess(img_path)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_text = decode(output)
    return row["text"], pred_text

total = len(df)
correct = 0
char_correct = 0
char_total = 0
wrong = []
confusion = defaultdict(int)

start_time = time.time()

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(worker, [row for _, row in df.iterrows()]))

for gt_text, pred_text in results:
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

elapsed = time.time() - start_time

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