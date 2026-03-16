import argparse
import os
import collections
import uuid
from typing import Dict, Tuple, List

import cv2
import numpy as np


def load_model(prototxt_path: str, weights_path: str, name: str = "model"):
    """Try to load a Caffe model; return None if missing or invalid."""
    if not prototxt_path or not weights_path:
        print(f"Warning: {name} paths not provided; running with dummy predictions.")
        return None

    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        print(f"Warning: {name} files not found; running with dummy predictions.")
        return None

    try:
        return cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    except Exception as e:
        print(f"Warning: Failed to load {name} model ({e}); running with dummy predictions.")
        return None


def preprocess_face(face):
    blob = cv2.dnn.blobFromImage(
        face,
        scalefactor=1.0,
        size=(227, 227),  # Fixed to standard model input size
        mean=(103.94, 116.78, 123.68),
        swapRB=True,
        crop=False,
    )
    return blob


def predict_age_gender(face, age_net, gender_net):
    age_ranges = [
        "(0-2)",
        "(4-6)",
        "(8-12)",
        "(15-20)",
        "(25-32)",
        "(38-43)",
        "(48-53)",
        "(60-100)",
    ]

    if age_net is None or gender_net is None:
        # Dummy prediction when models are missing or failed to load.
        gender = np.random.choice(["Male", "Female"])
        age = np.random.choice(age_ranges)
        return gender, age

    blob = preprocess_face(face)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = "Male" if gender_preds[0].argmax() == 0 else "Female"

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_ranges[age_preds[0].argmax()]

    return gender, age


def verify_predictions(predicted_gender, predicted_age, face_index):
    print(f"Face {face_index + 1}: Predicted Gender: {predicted_gender}, Age: {predicted_age}")
    correct_gender = input("Enter correct gender (Male/Female): ").strip().capitalize()
    correct_age = input("Enter correct age range (e.g., (25-32)): ").strip()
    gender_correct = predicted_gender == correct_gender
    age_correct = predicted_age == correct_age
    print(f"Gender correct: {gender_correct}, Age correct: {age_correct}")
    return gender_correct, age_correct


def find_faces(image):
    # Use OpenCV's default DNN face detector for best compatibility.
    # This will work if you have opencv >= 4.5 and the model files are installed.
    # We'll fall back to Haar cascades if the DNN face detector isn't available.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


def annotate(image, faces, age_gender):
    for (x, y, w, h), (gender, age) in zip(faces, age_gender):
        label = f"{gender}, {age}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


def create_tracker(buffer_size=5):
    return {
        'next_id': 0,
        'tracks': {},
        'buffer_size': buffer_size
    }

def update_tracks(tracker, faces, predictions):
    current_centroids = []
    for (x, y, w, h), (gender, age) in zip(faces, predictions):
        cx = x + w // 2
        cy = y + h // 2
        current_centroids.append((cx, cy, (x, y, w, h), (gender, age)))

    updated_tracks = {}
    used = set()
    for tid, track in tracker['tracks'].items():
        best_dist = float('inf')
        best_match = None
        for i, (cx, cy, _, _) in enumerate(current_centroids):
            if i in used: continue
            dist = ((cx - track['cx'])**2 + (cy - track['cy'])**2)**0.5
            if dist < best_dist and dist < 100:  # Threshold for matching
                best_dist = dist
                best_match = i
        if best_match is not None:
            i = best_match
            used.add(i)
            track['history_gender'].append(current_centroids[i][3][0])
            track['history_age'].append(current_centroids[i][3][1])
            if len(track['history_gender']) > tracker['buffer_size']:
                track['history_gender'].popleft()
                track['history_age'].popleft()
            track['cx'], track['cy'] = current_centroids[i][:2]
            track['bbox'] = current_centroids[i][2]
            updated_tracks[tid] = track

    for i, (cx, cy, bbox, pred) in enumerate(current_centroids):
        if i in used: continue
        tid = tracker['next_id']
        tracker['next_id'] += 1
        updated_tracks[tid] = {
            'cx': cx, 'cy': cy, 'bbox': bbox,
            'history_gender': collections.deque([pred[0]], maxlen=tracker['buffer_size']),
            'history_age': collections.deque([pred[1]], maxlen=tracker['buffer_size'])
        }

    tracker['tracks'] = updated_tracks
    return tracker

def get_smoothed_labels(tracker):
    labels = []
    bboxes = []
    for track in tracker['tracks'].values():
        genders = list(track['history_gender'])
        ages = list(track['history_age'])
        smoothed_gender = max(set(genders), key=genders.count)
        smoothed_age = max(set(ages), key=ages.count)
        labels.append((smoothed_gender, smoothed_age))
        bboxes.append(track['bbox'])
    return bboxes, labels

def main():
    parser = argparse.ArgumentParser(description="Run real-time age & gender detection using webcam.")
    parser.add_argument(
        "--age-prototxt",
        default="age_deploy.prototxt",
        help="Path to age deploy prototxt file.",
    )
    parser.add_argument(
        "--age-model",
        default="age_net.caffemodel",
        help="Path to age model weights (caffemodel).",
    )
    parser.add_argument(
        "--gender-prototxt",
        default="gender_deploy.prototxt",
        help="Path to gender deploy prototxt file.",
    )
    parser.add_argument(
        "--gender-model",
        default="gender_net.caffemodel",
        help="Path to gender model weights (caffemodel).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification mode to manually check predictions.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5,
        help="Number of frames for smoothing buffer.",
    )

    args = parser.parse_args()

    age_net = load_model(args.age_prototxt, args.age_model)
    gender_net = load_model(args.gender_prototxt, args.gender_model)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Starting real-time gender and age detection. Press 'q' to quit.")

    tracker = create_tracker(args.buffer_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = find_faces(frame)
        predictions = []
        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            predictions.append(predict_age_gender(face, age_net, gender_net))

        tracker = update_tracks(tracker, faces, predictions)

        tracked_bboxes, smoothed_labels = get_smoothed_labels(tracker)

        if args.verify:
            for i, (gender, age) in enumerate(smoothed_labels):
                verify_predictions(gender, age, i)

        annotate(frame, tracked_bboxes, smoothed_labels)
        cv2.imshow("Real-time Age & Gender Detection (Smoothed)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
