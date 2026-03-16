import unittest

import numpy as np

from detect_age_gender import FaceInsight, AGE_BUCKETS, GENDER_LABELS


class TestFaceInsight(unittest.TestCase):
    """Small unit test harness for FaceInsight.

    These tests avoid requiring the real Caffe models or a webcam. They exercise
    the internal prediction logic and smoothing behavior in dummy mode.
    """

    def setUp(self):
        # Use intentionally invalid model paths to force dummy mode.
        self.engine = FaceInsight(
            age_prototxt="missing.prototxt",
            age_model="missing.caffemodel",
            gender_prototxt="missing.prototxt",
            gender_model="missing.caffemodel",
            buffer_size=3,
            confidence_threshold=1.0,  # ensure dummy predictions are returned unchanged
        )

    def test_dummy_prediction_returns_known_labels(self):
        # Create a dummy image; exact content doesn't matter in dummy mode.
        dummy_image = np.zeros((227, 227, 3), dtype=np.uint8)
        gender, age = self.engine._predict_single(dummy_image)
        self.assertIn(gender, GENDER_LABELS)
        self.assertIn(age, AGE_BUCKETS)

    def test_smoothing_uses_majority_vote(self):
        # Simulate a track with flickering predictions. Majority should win.
        self.engine._tracker = self.engine._create_tracker(buffer_size=3)
        self.engine._tracker["tracks"][0] = {
            "cx": 0,
            "cy": 0,
            "bbox": (0, 0, 10, 10),
            "history_gender": collections.deque(["Male", "Female", "Male"], maxlen=3),
            "history_age": collections.deque(["(25-32)", "(25-32)", "(38-43)"], maxlen=3),
        }

        bboxes, labels = self.engine._get_smoothed_labels()
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(labels[0], ("Male", "(25-32)"))


if __name__ == "__main__":
    unittest.main()
