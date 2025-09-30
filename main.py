from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from PIL import Image
from io import BytesIO
from queue import Queue
from threading import Thread, Lock, Condition
import uvicorn
import onnxruntime as ort
import numpy as np
import string
import uuid
from torchvision import transforms

app = FastAPI()
task_queue = Queue()
result_store = {}
result_lock = Lock()

charset = string.ascii_uppercase + string.ascii_lowercase + string.digits
device = "cpu"

ort_session = ort.InferenceSession("captcha_crnn.onnx", providers=["CPUExecutionProvider"])

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def ctc_decode(output, charset):
    output = output.argmax(2)
    decoded = []
    for seq in output:
        chars = []
        prev_idx = 0
        for idx in seq:
            idx = int(idx)
            if idx != 0 and idx != prev_idx:
                chars.append(charset[idx-1])
            prev_idx = idx
        decoded.append("".join(chars))
    return decoded[0]

def worker():
    while True:
        task_id, image_bytes = task_queue.get()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("L")
            img_tensor = transform(image).unsqueeze(0).numpy()
            output = ort_session.run(None, {"input": img_tensor})[0]
            text = ctc_decode(output, charset)
        except Exception as e:
            text = f"ERROR: {str(e)}"
        condition = result_store[task_id]["condition"]
        with condition:
            result_store[task_id]["result"] = text
            condition.notify()
        task_queue.task_done()

for _ in range(2):
    Thread(target=worker, daemon=True).start()

@app.post("/captcha", response_class=PlainTextResponse)
async def solve_captcha(file: UploadFile = File(...)):
    image_bytes = await file.read()
    task_id = str(uuid.uuid4())
    condition = Condition()
    with result_lock:
        result_store[task_id] = {
            "condition": condition,
            "result": None,
        }
    task_queue.put((task_id, image_bytes))
    with condition:
        condition.wait()
    with result_lock:
        result = result_store.pop(task_id)["result"]
    return result

@app.get("/ping")
async def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)