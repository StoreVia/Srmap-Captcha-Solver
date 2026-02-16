import subprocess
import sys

ONNX_MODEL = "../models/captcha_model.onnx"

def main():
    try:
        result = subprocess.run(
            ["onnx2tf", "-i", ONNX_MODEL],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        print("Conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Conversion failed.")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()