import importlib
import io
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from backend.app import detector as detector_module


class StubDetector:
    names = {0: "test"}

    def predict(self, image, conf=0.25, iou=0.5):
        return {"width": image.width, "height": image.height, "detections": []}


with patch.object(detector_module, "Detector", StubDetector):
    api = importlib.import_module("backend.app.main")


def png_bytes():
    image = Image.new("RGB", (1, 1), (255, 0, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class TrackingBytesIO(io.BytesIO):
    last_read_size = None

    def read(self, size=-1):
        self.last_read_size = size
        return super().read(size)


class DetectEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_max_bytes = api.MAX_BYTES
        self.original_max_pixels = api.MAX_PIXELS

    def tearDown(self):
        api.MAX_BYTES = self.original_max_bytes
        api.MAX_PIXELS = self.original_max_pixels

    async def test_detect_rejects_oversized_upload_with_bounded_read(self):
        from fastapi import UploadFile

        api.MAX_BYTES = 8
        stream = TrackingBytesIO(b"x" * 32)
        upload = UploadFile(filename="large.png", file=stream)

        with self.assertRaises(api.HTTPException) as raised:
            await api.detect(upload, conf=0.25, iou=0.5)

        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(stream.last_read_size, api.MAX_BYTES + 1)

    async def test_detect_rejects_oversized_dimensions_before_loading_image(self):
        from fastapi import UploadFile

        api.MAX_BYTES = 32
        api.MAX_PIXELS = 10
        image = Mock(width=4, height=3)
        upload = UploadFile(filename="large.png", file=io.BytesIO(b"image"))

        with patch.object(api.Image, "open", return_value=image):
            with self.assertRaises(api.HTTPException) as raised:
                await api.detect(upload, conf=0.25, iou=0.5)

        self.assertEqual(raised.exception.status_code, 413)
        image.load.assert_not_called()

    async def test_detect_accepts_valid_image_at_size_limit(self):
        from fastapi import UploadFile

        data = png_bytes()
        api.MAX_BYTES = len(data)
        api.MAX_PIXELS = 1
        upload = UploadFile(filename="small.png", file=io.BytesIO(data))
        result = await api.detect(upload, conf=0.25, iou=0.5)

        self.assertEqual(result, {"width": 1, "height": 1, "detections": []})

    async def test_detect_rejects_invalid_image(self):
        from fastapi import UploadFile

        api.MAX_BYTES = 32
        upload = UploadFile(filename="invalid.png", file=io.BytesIO(b"not an image"))

        with self.assertRaises(api.HTTPException) as raised:
            await api.detect(upload, conf=0.25, iou=0.5)

        self.assertEqual(raised.exception.status_code, 400)


class HealthEndpointTests(unittest.TestCase):
    def test_health_reports_status_and_detector_classes(self):
        result = api.health()

        self.assertEqual(result, {"status": "ok", "classes": StubDetector.names})


class DetectParameterTests(unittest.TestCase):
    def test_openapi_documents_confidence_and_iou_bounds(self):
        parameters = api.app.openapi()["paths"]["/api/detect"]["post"]["parameters"]
        query_parameters = {parameter["name"]: parameter for parameter in parameters}

        self.assertEqual(
            query_parameters["conf"]["schema"],
            {
                "type": "number",
                "maximum": 0.99,
                "minimum": 0.01,
                "default": 0.25,
                "title": "Conf",
            },
        )
        self.assertEqual(
            query_parameters["iou"]["schema"],
            {
                "type": "number",
                "maximum": 0.99,
                "minimum": 0.1,
                "default": 0.5,
                "title": "Iou",
            },
        )


if __name__ == "__main__":
    unittest.main()
