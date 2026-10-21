from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn

from model_engine import engine, ROAD_MULTIPLIERS, FLOOD_PRONE
from pdf_generator import create_pdf_bytes

app = FastAPI(title="Urbanscapes-LK Model Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    suburb: str = Field(..., example="athurugiriya")
    land_size: float = Field(12.0, ge=1.0, le=200.0)
    land_type: str = Field("Residential")
    road_width: str = Field("15 ft")
    target_year: int = Field(2026, ge=2026, le=2035)
    lat: float = Field(6.9022)
    lon: float = Field(79.9172)

class PdfRequest(BaseModel):
    parcel_info: Dict[str, Any]
    valuation_info: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "service": "Urbanscapes-LK Python Model Engine"}

@app.get("/metadata")
def get_metadata():
    return {
        "suburbs": engine.suburbs,
        "land_types": engine.land_types,
        "road_widths": list(ROAD_MULTIPLIERS.keys()),
        "flood_prone": FLOOD_PRONE,
        "default_coords": {"lat": 6.9022, "lon": 79.9172},
        "fort_coords": {"lat": 6.9344, "lon": 79.8428}
    }

@app.post("/predict")
def predict_valuation(req: PredictRequest):
    try:
        result = engine.predict(
            suburb=req.suburb,
            land_size=req.land_size,
            land_type=req.land_type,
            road_width=req.road_width,
            target_year=req.target_year,
            lat=req.lat,
            lon=req.lon
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/generate-pdf")
def generate_pdf(req: PdfRequest):
    try:
        import base64
        pdf_bytes = create_pdf_bytes(req.parcel_info, req.valuation_info)
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        return {
            "success": True,
            "pdf_base64": b64_pdf
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5001, reload=False)
