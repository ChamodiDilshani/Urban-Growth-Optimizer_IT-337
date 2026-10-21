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

    def test_predict_keeps_overlapping_detections_from_different_classes(self):
        class Session:
            def run(self, _outputs, _inputs):
                output = np.zeros((1, 8, 8400), dtype=np.float32)
                output[0, :4, :3] = np.array(
                    [[320, 320, 320], [320, 320, 320], [100, 100, 100], [100, 100, 100]],
                    dtype=np.float32,
                )
                output[0, 4, :2] = [0.9, 0.7]
                output[0, 5, 1] = 0.8
                return [output]

        self.detector.session = Session()
        self.detector.input_name = "images"
        self.detector.names = {0: "drain", 1: "manhole"}

        result = self.detector.predict(Image.new("RGB", (IMGSZ, IMGSZ)))

        self.assertEqual([item["class_id"] for item in result["detections"]], [0, 1])
        self.assertEqual(
            [item["label"] for item in result["detections"]], ["drain", "manhole"]
        )

    def test_predict_filters_confidence_and_restores_original_image_coordinates(self):
        class Session:
            def run(self, _outputs, _inputs):
                output = np.zeros((1, 8, 8400), dtype=np.float32)
                output[0, :4, :2] = np.array(
                    [[320, 320], [320, 320], [200, 200], [200, 200]],
                    dtype=np.float32,
                )
                output[0, 6, 0] = 0.9
                output[0, 5, 1] = 0.4
                return [output]

        self.detector.session = Session()
        self.detector.input_name = "images"
        self.detector.names = {1: "drain", 2: "manhole"}

        result = self.detector.predict(
            Image.new("RGB", (320, 160)), conf=0.5, iou=0.5
        )

        self.assertEqual(result["width"], 320)
        self.assertEqual(result["height"], 160)
        self.assertEqual(len(result["detections"]), 1)
        detection = result["detections"][0]
        self.assertEqual(detection["class_id"], 2)
        self.assertEqual(detection["label"], "manhole")
        self.assertEqual(detection["confidence"], 0.9)
        self.assertEqual(detection["box"], [110.0, 30.0, 210.0, 130.0])


if __name__ == "__main__":
    unittest.main()
