import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.datasets import mnist
from keras.models import load_model

# Load MNIST dataset and pre-trained model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = load_model('models/mnistCNN.h5.keras', compile=False)  # Ignore optimizer state

def get_numbers(y_pred):
    final_number = str(np.argmax(y_pred))
    per = round((np.max(y_pred) * 100), 2)
    return final_number, per

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition")
        self.root.geometry("800x600")

        self.video = cv2.VideoCapture(0)

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.btn_capture = tk.Button(root, text="Capture", command=self.capture_image)
        self.btn_capture.pack(pady=10)

        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack(pady=10)

        self.label_result = tk.Label(root, text="Predicted Value: None", font=("Helvetica", 14))
        self.label_result.pack(pady=10)

        self.update_frame()

    def update_frame(self):
        check, frame = self.video.read()
        if check:
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.img = Image.fromarray(self.cv2image)
            self.imgtk = ImageTk.PhotoImage(image=self.img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)
        self.root.after(10, self.update_frame)

    def capture_image(self):
        check, frame = self.video.read()
        if check:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", ".png"), ("All files", ".*")])
            if file_path:
                cv2.imwrite(file_path, frame)
                self.process_image(file_path)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG files", ".png"), ("All files", ".*")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        capture_img = cv2.imread(file_path)
        if capture_img is None:
            messagebox.showerror("Error", "Unable to read the image file")
        else:
            img2 = capture_img.copy()

            # Image processing for prediction
            img_gray = cv2.cvtColor(capture_img, cv2.COLOR_BGR2GRAY)
            img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
            ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(thresh, kernel, iterations=1)
            edged = cv2.Canny(dilation, 50, 250)

            # Find contours
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            num_list = []
            for c in contours:
                if cv2.contourArea(c) > 500:  # Adjusted area threshold
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 3)

                    new_img = thresh[y:y + h, x:x + w]
                    new_img2 = cv2.resize(new_img, (28, 28))
                    # Normalize the image
                    new_img2 = new_img2.astype('float32') / 255
                    im2arr = new_img2.reshape(1, 28, 28, 1)

                    y_pred = model.predict(im2arr)
                    print(f'Prediction raw output: {y_pred}')

                    num, _ = get_numbers(y_pred)
                    num_list.append(num)
                    cv2.putText(img2, f'[{num}]', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            num_str = ' '.join(num_list)
            if num_str:
                y_p = f'Predicted Value is {num_str}'
                print(y_p)
                self.label_result.config(text=y_p)

            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            top = tk.Toplevel(self.root)
            top.title("Captured Frame")
            canvas2 = tk.Canvas(top, width=img2.width, height=img2.height)
            canvas2.pack()
            canvas2.create_image(0, 0, anchor=tk.NW, image=imgtk2)
            top.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
