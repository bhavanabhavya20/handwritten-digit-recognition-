import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from keras.models import load_model

# Load MNIST dataset and pre-trained model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = load_model('models/mnistCNN.h5.keras', compile=False)  # Ignore optimizer state

def get_numbers(y_pred):
    for number, per in enumerate(y_pred[0]):
        if per != 0:
            final_number = str(int(number))
            per = round((per * 100), 2)
            return final_number, per

video = cv2.VideoCapture(0)
if video.isOpened():
    while True:
        check, img = video.read()
        cv2.imshow("Frame", img)

        # Preprocessing for display
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("Frame thresh", thresh)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imshow("Frame dilation", dilation)

        edged = cv2.Canny(dilation, 50, 250)
        cv2.imshow("Frame edged", edged)

        key = cv2.waitKey(1)

        if key == 27:  # Exit on ESC key
            break
        elif key & 0xFF == ord('c'):  # Capture on 'c' key
            cv2.imwrite('B:/output.png', img)
            capture_img = cv2.imread('B:/output.png')

            if capture_img is None:
                print("Error: Unable to read the image file")
            else:
                img2 = capture_img.copy()

                # Image processing for prediction
                img_gray = cv2.cvtColor(capture_img, cv2.COLOR_RGB2GRAY)
                img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
                ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((5, 5), np.uint8)
                dilation = cv2.dilate(thresh, kernel, iterations=1)
                edged = cv2.Canny(dilation, 50, 250)

                # Find contours
                contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                num_list = []
                if len(contours) > 0:
                    for c in contours:
                        if cv2.contourArea(c) > 2500:
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 3)

                            new_img = thresh[y:y + h, x:x + w]
                            new_img2 = cv2.resize(new_img, (28, 28))
                            im2arr = np.array(new_img2).reshape(1, 28, 28, 1)

                            y_pred = model.predict(im2arr)
                            print(y_pred)

                            num, _ = get_numbers(y_pred)
                            num_list.append(str(int(num)))
                            cv2.putText(img2, f'[{num}]', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

                    num_str = ' '.join(num_list)
                    if num_str:
                        y_p = f'Predicted Value is {num_str}'
                        print(y_p)
                        cv2.putText(img2, y_p, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("Capture Frame", img2)

video.release()
cv2.destroyAllWindows()
