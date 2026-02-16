import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import string

chars = string.ascii_uppercase + string.digits + "_"
idx_to_char = {i: c for i, c in enumerate(chars)}

session = ort.InferenceSession("./models/captcha_model.onnx")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

img = Image.open("captcha.png").convert("L")
img = img.crop((0, 0, 120, 25))
img = transform(img).unsqueeze(0).numpy()

outputs = session.run(None, {"input": img})
pred = np.argmax(outputs[0], axis=2)[0]
text = "".join([idx_to_char[i] for i in pred]).replace("_", "")
print("Prediction:", text)