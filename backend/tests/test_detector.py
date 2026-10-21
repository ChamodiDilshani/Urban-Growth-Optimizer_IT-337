import unittest

import numpy as np
from PIL import Image

from backend.app.detector import Detector, IMGSZ


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = Detector.__new__(Detector)

    def test_preprocess_letterboxes_image_and_returns_transform(self):
        image = Image.new("RGB", (320, 160), (255, 0, 0))

        tensor, scale, pad_x, pad_y = self.detector._preprocess(image)

        self.assertEqual(tensor.shape, (1, 3, IMGSZ, IMGSZ))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual((scale, pad_x, pad_y), (2.0, 0, 160))
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255)
        np.testing.assert_allclose(tensor[0, :, 320, 320], (1.0, 0.0, 0.0))

    def test_nms_keeps_highest_scoring_overlapping_box_and_separate_box(self):
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        kept = Detector._nms(boxes, scores, 0.5)

        self.assertEqual(kept, [0, 2])

    def test_nms_returns_empty_list_for_no_boxes(self):
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)

        self.assertEqual(Detector._nms(boxes, scores, 0.5), [])


if __name__ == "__main__":
    unittest.main()
