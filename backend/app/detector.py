"""YOLO11 ONNX inference (letterbox -> model -> confidence filter -> NMS)."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
IMGSZ = 640


class Detector:
    def __init__(self):
        self.session = ort.InferenceSession(
            str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        with open(MODEL_DIR / "label.json", encoding="utf-8") as f:
            self.names = {int(k): v for k, v in json.load(f).items()}

    def _preprocess(self, img: Image.Image):
        w, h = img.size
        r = min(IMGSZ / w, IMGSZ / h)
        nw, nh = round(w * r), round(h * r)
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
        px, py = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
        canvas.paste(resized, (px, py))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None], r, px, py

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-9)
            order = rest[iou <= iou_thr]
        return keep

    def predict(self, img: Image.Image, conf=0.25, iou=0.5):
        img = img.convert("RGB")
        W, H = img.size
        x, r, px, py = self._preprocess(img)
        out = self.session.run(None, {self.input_name: x})[0][0].T  # (8400, 4+nc)
        cls_scores = out[:, 4:]
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores.max(1)
        m = scores >= conf
        out, cls_ids, scores = out[m], cls_ids[m], scores[m]
        if not len(scores):
            return {"width": W, "height": H, "detections": []}

        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / r).clip(0, W)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / r).clip(0, H)

        # class-aware NMS via coordinate offset
        keep = self._nms(boxes + cls_ids[:, None] * 10000.0, scores, iou)
        dets = [
            {
                "class_id": int(cls_ids[i]),
                "label": self.names[int(cls_ids[i])],
                "confidence": round(float(scores[i]), 4),
                "box": [round(float(v), 1) for v in boxes[i]],  # x1,y1,x2,y2 in original px
            }
            for i in keep
        ]
        return {"width": W, "height": H, "detections": dets}
