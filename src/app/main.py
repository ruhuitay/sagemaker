"""Draw a digit with your mouse and send it to a Triton inference server."""

import tkinter as tk

import numpy as np
import requests
from PIL import Image, ImageDraw


TRITON_URL = "http://localhost:8000/v2/models/mnist/infer"


class DigitCanvas:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a digit (press Enter to predict, Escape to clear)")

        self.canvas_size = 280
        self.canvas = tk.Canvas(
            self.root, width=self.canvas_size, height=self.canvas_size, bg="black"
        )
        self.canvas.pack()

        # PIL image to draw on (matches canvas)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.root.bind("<Return>", lambda e: self.predict())
        self.root.bind("<Escape>", lambda e: self.clear())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        self.label = tk.Label(
            self.root, text="Draw a digit | Enter = predict | Escape = clear"
        )
        self.label.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        predict_btn = tk.Button(button_frame, text="Predict", command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear)
        clear_btn.pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 24))
        self.result_label.pack()

        self.root.mainloop()

    def paint(self, event):
        r = 12  # brush radius
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        """Clear the canvas and reset for a new drawing."""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def preprocess(self) -> list:
        """Resize to 28x28 and normalize to [0, 1] float32 tensor shape [1, 1, 28, 28]."""
        small = self.image.resize((28, 28), Image.LANCZOS)
        arr = np.array(small, dtype=np.float32) / 255.0
        # Shape: [1, 1, 28, 28] - batch, channel, height, width
        tensor = arr.reshape(1, 1, 28, 28)
        return tensor.flatten().tolist()

    def predict(self):
        """Send the drawn image to the Triton server and display the result."""
        data = self.preprocess()

        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 1, 28, 28],
                    "datatype": "FP32",
                    "data": data,
                }
            ],
        }

        try:
            response = requests.post(TRITON_URL, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            output_data = result["outputs"][0]["data"]
            predicted_digit = int(np.argmax(output_data))
            confidence = float(np.max(self._softmax(output_data))) * 100
            self.result_label.config(
                text=f"Prediction: {predicted_digit} ({confidence:.1f}%)"
            )
            print(f"Predicted: {predicted_digit} (confidence: {confidence:.1f}%)")
        except requests.exceptions.ConnectionError:
            self.result_label.config(text="Error: Cannot connect to Triton server")
        except requests.exceptions.RequestException as e:
            self.result_label.config(text=f"Error: {e}")
            print(f"Request failed: {e}")

    @staticmethod
    def _softmax(logits):
        """Compute softmax probabilities from raw logits."""
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()


if __name__ == "__main__":
    DigitCanvas()
