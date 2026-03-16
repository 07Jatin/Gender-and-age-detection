"""Real-time Age & Gender detection with a structured, production-friendly API.

This module is designed to be used either as a standalone script or imported into a
larger application (e.g., a GUI or service). It focuses on readability, maintain-ability,
and a small set of practical defaults.

Notes:
- The Caffe models were trained with 227x227 inputs and mean subtraction; we keep
  that behavior here and avoid scaling to [0, 1] to match training-time preprocessing.
- Face tracking is intentionally simplified (centroid matching) to reduce flicker.

TODO:
- Add robust multi-face tracking (ID persistence across occlusions).
- Add low-light preprocessing (CLAHE or denoising) to improve low-light accuracy.
- Add batch evaluation mode + metrics logging for offline dataset evaluation.
- Expose a callback API to allow embedding into a UI thread (e.g., Qt, Tkinter).
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger("FaceInsight")
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)


AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]

GENDER_LABELS = ["Male", "Female"]


class FaceInsight:
    """Encapsulates face detection, prediction, and smoothing logic."""

    def __init__(
        self,
        age_prototxt: str,
        age_model: str,
        gender_prototxt: str,
        gender_model: str,
        buffer_size: int = 5,
        confidence_threshold: float = 0.6,
    ):
        self.age_net = self._load_model(age_prototxt, age_model, "age")
        self.gender_net = self._load_model(gender_prototxt, gender_model, "gender")
        self._use_dummy_models = self.age_net is None or self.gender_net is None

        if self._use_dummy_models:
            logger.warning(
                "One or more models failed to load. Running in dummy mode (randomized outputs)."
            )

        self.buffer_size = max(1, buffer_size)
        self.confidence_threshold = confidence_threshold
        self._tracker = self._create_tracker(self.buffer_size)

        # Haar cascade is a lightweight fallback for face detection. You can replace
        # this with a DNN-based detector if higher accuracy is required.
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _load_model(self, prototxt_path: str, weights_path: str, name: str):
        if not prototxt_path or not weights_path:
            logger.warning("No paths provided for %s model; will use dummy predictions.", name)
            return None

        if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
            logger.warning(
                "Unable to find %s model files (%s, %s); will use dummy predictions.",
                name,
                prototxt_path,
                weights_path,
            )
            return None

        try:
            net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
            logger.info("Loaded %s model from %s", name, weights_path)
            return net
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load %s model: %s", name, exc)
            return None

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        # The models expect the image preprocessing used during training:
        #   1) Resize to 227x227
        #   2) Convert to BGR (OpenCV default)
        #   3) Subtract mean values (as in training)
        #   4) Keep scale factor 1.0 (no division by 255)
        return cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0,
            size=(227, 227),
            mean=(103.94, 116.78, 123.68),
            swapRB=True,
            crop=False,
        )

    def _predict_single(self, face: np.ndarray) -> Tuple[str, str]:
        if self._use_dummy_models:
            # Dummy mode makes it easier to work on UI elements without the models.
            return np.random.choice(GENDER_LABELS), np.random.choice(AGE_BUCKETS)

        blob = self._preprocess(face)

        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()[0]
        gender_idx = int(np.argmax(gender_preds))
        gender_confidence = float(gender_preds[gender_idx])
        gender_label = GENDER_LABELS[gender_idx]

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()[0]
        age_idx = int(np.argmax(age_preds))
        age_confidence = float(age_preds[age_idx])
        age_label = AGE_BUCKETS[age_idx]

        if gender_confidence < self.confidence_threshold:
            logger.debug(
                "Low gender confidence %.2f; using 'Unknown' instead of %s.",
                gender_confidence,
                gender_label,
            )
            gender_label = "Unknown"

        if age_confidence < self.confidence_threshold:
            logger.debug(
                "Low age confidence %.2f; using 'Unknown' instead of %s.",
                age_confidence,
                age_label,
            )
            age_label = "Unknown"

        return gender_label, age_label

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return faces

    @staticmethod
    def _create_tracker(buffer_size: int) -> Dict:
        return {"next_id": 0, "tracks": {}, "buffer_size": buffer_size}

    def _update_tracks(
        self, faces: List[Tuple[int, int, int, int]], predictions: List[Tuple[str, str]]
    ) -> None:
        current = []
        for (x, y, w, h), (gender, age) in zip(faces, predictions):
            current.append((x + w // 2, y + h // 2, (x, y, w, h), (gender, age)))

        updated = {}
        used = set()

        for tid, track in self._tracker["tracks"].items():
            best_idx = None
            best_dist = float("inf")

            for i, (cx, cy, _, _) in enumerate(current):
                if i in used:
                    continue
                dist = ((cx - track["cx"]) ** 2 + (cy - track["cy"]) ** 2) ** 0.5
                if dist < best_dist and dist < 100:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None:
                used.add(best_idx)
                cx, cy, bbox, (gender, age) = current[best_idx]
                track["cx"] = cx
                track["cy"] = cy
                track["bbox"] = bbox
                track["history_gender"].append(gender)
                track["history_age"].append(age)
                updated[tid] = track

        for i, (cx, cy, bbox, (gender, age)) in enumerate(current):
            if i in used:
                continue
            tid = self._tracker["next_id"]
            self._tracker["next_id"] += 1
            updated[tid] = {
                "cx": cx,
                "cy": cy,
                "bbox": bbox,
                "history_gender": collections.deque([gender], maxlen=self.buffer_size),
                "history_age": collections.deque([age], maxlen=self.buffer_size),
            }

        self._tracker["tracks"] = updated

    def _get_smoothed_labels(
        self,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[str, str]]]:
        bboxes: List[Tuple[int, int, int, int]] = []
        labels: List[Tuple[str, str]] = []

        for track in self._tracker["tracks"].values():
            genders = list(track["history_gender"])
            ages = list(track["history_age"])
            smoothed_gender = max(set(genders), key=genders.count)
            smoothed_age = max(set(ages), key=ages.count)
            bboxes.append(track["bbox"])
            labels.append((smoothed_gender, smoothed_age))

        return bboxes, labels

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        labels: List[Tuple[str, str]],
    ) -> None:
        for (x, y, w, h), (gender, age) in zip(bboxes, labels):
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    def run(
        self,
        camera_index: int = 0,
        verify: bool = False,
        window_name: str = "Real-time Age & Gender Detection",
    ) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(
                "Unable to open camera index %s. Please ensure a webcam is connected.",
                camera_index,
            )
            raise RuntimeError("Could not open camera")

        logger.info("Starting live detection (press 'q' to quit)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame grab failed; stopping.")
                    break

                faces = self._detect_faces(frame)
                predictions = [
                    self._predict_single(frame[y : y + h, x : x + w]) for (x, y, w, h) in faces
                ]

                self._update_tracks(faces, predictions)
                bboxes, labels = self._get_smoothed_labels()

                if verify:
                    for idx, (gender, age) in enumerate(labels):
                        self._verify_predictions(gender, age, idx)

                self._annotate(frame, bboxes, labels)
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _verify_predictions(
        self, predicted_gender: str, predicted_age: str, face_index: int
    ) -> Tuple[bool, bool]:
        logger.info(
            "Face %s: predicted Gender=%s Age=%s",
            face_index + 1,
            predicted_gender,
            predicted_age,
        )
        correct_gender = input("Enter correct gender (Male/Female): ").strip().capitalize()
        correct_age = input("Enter correct age range (e.g., (25-32)): ").strip()
        gender_correct = predicted_gender == correct_gender
        age_correct = predicted_age == correct_age
        logger.info("Verification results: gender=%s age=%s", gender_correct, age_correct)
        return gender_correct, age_correct


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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum softmax confidence for labels to be considered valid.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (0 is default).",
    )

    args = parser.parse_args()

    engine = FaceInsight(
        age_prototxt=args.age_prototxt,
        age_model=args.age_model,
        gender_prototxt=args.gender_prototxt,
        gender_model=args.gender_model,
        buffer_size=args.buffer_size,
        confidence_threshold=args.confidence_threshold,
    )

    try:
        engine.run(camera_index=args.camera_index, verify=args.verify)
    except Exception as ex:  # pragma: no cover
        logger.exception("Unexpected error during runtime: %s", ex)
        sys.exit(1)


if __name__ == "__main__":
    main()
