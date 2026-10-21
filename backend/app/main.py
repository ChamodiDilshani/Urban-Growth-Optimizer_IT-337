import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .detector import Detector

app = FastAPI(title="Manhole & Drain Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = Detector()
MAX_BYTES = 15 * 1024 * 1024


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": detector.names}


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    iou: float = Query(0.5, ge=0.1, le=0.99),
):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a valid image")
    return detector.predict(img, conf=conf, iou=iou)


# Serve the built React app when present (production / Docker). Registered last
# so the /api routes above take priority.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
