"""Simple sample driver that shows face detection on `sample.jpg`.

This file is not intended to be run as a unit test; it is a small manual demo.
"""

import cv2


def run_sample():
    image = cv2.imread("sample.jpg")
    if image is None:
        raise FileNotFoundError("sample.jpg not found")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    print(f"Detected {len(faces)} faces in sample.jpg")
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Sample", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_sample()
