"""Draw a digit with your mouse and save it for prediction."""

import tkinter as tk
from PIL import Image, ImageDraw


class DigitCanvas:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a digit (press Enter or close to save)")

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
        self.root.bind("<Return>", lambda e: self.save_and_quit())
        self.root.protocol("WM_DELETE_WINDOW", self.save_and_quit)

        label = tk.Label(self.root, text="Draw a digit, then press Enter or close the window")
        label.pack()

        self.root.mainloop()

    def paint(self, event):
        r = 12  # brush radius
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def save_and_quit(self):
        # Resize to 28x28 (MNIST size)
        small = self.image.resize((28, 28), Image.LANCZOS)
        small.save("my_digit.png")
        print("Saved to my_digit.png (28x28 grayscale)")
        self.root.destroy()


if __name__ == "__main__":
    DigitCanvas()
