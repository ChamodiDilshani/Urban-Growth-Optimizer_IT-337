# Urban-Growth-Optimizer_IT-337

This project supports spatial growth prediction, unauthorized building detection,
land valuation, and infrastructure optimization for the Urban Development
Authority of Sri Lanka.

## Manhole and Drain Detection API

### Setup

From the repository root, install the backend dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

The detector expects these files:

- `backend\models\model.onnx`
- `backend\models\label.json`

The ONNX model is not tracked in this repository. Obtain and place the model at
the path above before starting the API.

### Run

From the repository root, start the development server with:

```powershell
uvicorn app.main:app --app-dir backend --reload
```

### Endpoints

- `GET /api/health` returns the API status and detector classes.
- `POST /api/detect` accepts an image as multipart form data in the `file` field.
  Optional `conf` and `iou` query parameters configure confidence and
  intersection-over-union thresholds.

### Tests

From the repository root, run the backend unit tests with:

```powershell
python -m unittest discover -s backend\tests -v
```
