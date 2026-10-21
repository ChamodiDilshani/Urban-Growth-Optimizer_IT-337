"""
FastAPI server for the illegal-land / encroachment detector.

Run:  uvicorn app.main:app --reload --port 8000
"""
import base64
import threading
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from . import pipeline as P

MAX_SIDE = 4096                 # refuse absurdly large uploads (memory guard)
MAX_BYTES = 30 * 1024 * 1024    # 30 MB per image

app = FastAPI(title="Illegal Land Detector API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- model state (loaded once at startup) ----
STATE = dict(model=None, source=None, error=None, threshold=0.5, metrics=None, device="cpu")
LOCK = threading.Lock()   # one inference at a time (keeps memory predictable)


@app.on_event("startup")
def _load():
    try:
        from .model import DEVICE, load_model, load_test_metrics, load_threshold
        STATE["device"] = DEVICE.type
        STATE["threshold"] = load_threshold()
        STATE["metrics"] = load_test_metrics()
        STATE["model"], STATE["source"] = load_model()
        print(f"[model] loaded {STATE['source']} on {DEVICE} | threshold={STATE['threshold']}")
    except Exception as e:  # keep the server up so the UI can show the error
        STATE["error"] = str(e)
        print("[model] FAILED:", e)


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


async def _read_image(upload: UploadFile, label: str):
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"The {label} image is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"The {label} image is larger than 30 MB.")
    img = P.decode_rgb(data)
    if img is None:
        raise HTTPException(400, f"Could not read the {label} image. Use PNG, JPG, TIFF or BMP.")
    if max(img.shape[:2]) > MAX_SIDE:
        raise HTTPException(413, f"The {label} image is over {MAX_SIDE}px on its longest side.")
    return img


@app.get("/api/health")
def health():
    return dict(
        ready=STATE["model"] is not None,
        error=STATE["error"],
        model_file=STATE["source"],
        device=STATE["device"],
        threshold=STATE["threshold"],
        test_metrics=STATE["metrics"],
        pixel_size_m=P.PIXEL_SIZE_M,
        min_area_px=P.MIN_AREA_PX,
        min_overlap_px=P.MIN_OVERLAP_PX,
    )


@app.post("/api/analyze")
async def analyze(
    before: UploadFile = File(..., description="T1 image (before)"),
    after: UploadFile = File(..., description="T2 image (after)"),
    zone: Optional[UploadFile] = File(None, description="T1 image with the protected zone painted red"),
    threshold: Optional[float] = Form(None),
    tta: bool = Form(True),
):
    if STATE["model"] is None:
        raise HTTPException(503, f"Model is not loaded: {STATE['error'] or 'still starting'}")

    before_img = await _read_image(before, "before")
    after_img = await _read_image(after, "after")
    zone_img = await _read_image(zone, "zone") if zone is not None and zone.filename else None

    thr = STATE["threshold"] if threshold is None else float(threshold)
    if not 0.05 <= thr <= 0.95:
        raise HTTPException(400, "Threshold must be between 0.05 and 0.95.")

    warnings = []
    if after_img.shape != before_img.shape:
        warnings.append(f"The after image ({after_img.shape[1]}x{after_img.shape[0]}) was resized to match "
                        f"the before image ({before_img.shape[1]}x{before_img.shape[0]}). "
                        "Results are only reliable when both images cover exactly the same area.")
    if zone_img is not None and zone_img.shape != before_img.shape:
        warnings.append("The zone image was resized to match the before image.")

    # heavy work runs in a worker thread so the server stays responsive
    return await run_in_threadpool(_run, before_img, after_img, zone_img, thr, tta, warnings)


def _run(before_img, after_img, zone_img, thr, tta, warnings):
    t0 = time.time()
    with LOCK:
        prob = P.predict_prob(STATE["model"], _device(), before_img, after_img, tta=tta)
    t_model = time.time() - t0

    # same as the notebook: outputs are drawn on the after image at the before image's size
    if after_img.shape != before_img.shape:
        import cv2
        after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))

    res = P.analyze(prob, before_img, zone_img, thr)
    if zone_img is not None and res["summary"]["zone_pct"] == 0:
        warnings.append("No red highlight was found in the zone image, so nothing could be checked against "
                        "a protected zone. Paint the protected area in solid red on a copy of the before image.")

    img_change = P.render_change(after_img, res)
    img_comp = P.render_compliance(after_img, res)

    return dict(
        summary=res["summary"],
        detections=res["rows"],
        geojson=res["geojson"],
        warnings=warnings,
        params=dict(threshold=thr, tta=tta, pixel_size_m=P.PIXEL_SIZE_M,
                    width=int(before_img.shape[1]), height=int(before_img.shape[0])),
        timing=dict(model_s=round(t_model, 2), total_s=round(time.time() - t0, 2)),
        images=dict(
            change_result=_data_url(P.encode_png_rgb(img_change)),
            compliance_result=_data_url(P.encode_png_rgb(img_comp)),
            change_overlay=_data_url(P.encode_png_rgba(P.change_overlay(res))),
            zone_overlay=_data_url(P.encode_png_rgba(P.zone_overlay(res["zone"]))),
            # browser-safe copies for the comparison viewer (aligned to the same size)
            before=_data_url(P.encode_jpg_rgb(before_img), "image/jpeg"),
            after=_data_url(P.encode_jpg_rgb(after_img), "image/jpeg"),
        ),
    )


def _device():
    from .model import DEVICE
    return DEVICE
