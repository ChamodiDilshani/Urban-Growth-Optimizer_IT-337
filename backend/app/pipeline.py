"""
Encroachment pipeline, ported from the notebook cells:
  predict_prob -> clean_mask -> extract_zone -> analyze -> render_change / render_compliance

Images are RGB uint8 numpy arrays (H, W, 3) throughout, exactly as in the notebook.
geopandas is replaced with plain dicts + shapely so the server stays light;
the per-object numbers are computed the same way.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

PIXEL_SIZE_M = 0.5            # LEVIR-CD+ ground sampling distance (~0.5 m/pixel)
PIX_AREA_M2 = PIXEL_SIZE_M ** 2
MIN_AREA_PX = 40              # ignore detected blobs smaller than this
MIN_OVERLAP_PX = 5            # minimum overlap with the zone to count as encroachment

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------- I/O helpers ----------------
def decode_rgb(data: bytes):
    """Uploaded file bytes -> RGB uint8 array (same as notebook read_rgb)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png_rgb(img):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def encode_jpg_rgb(img, quality=88):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def encode_png_rgba(img_rgba):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))
    return buf.tobytes()


def to_tensor(img):
    import torch
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


# ---------------- 1. Change prediction ----------------
def predict_prob(model, device, before, after, tta=True):
    """Returns change probability map. Pads to a multiple of 32; optional flip TTA."""
    import torch
    import torch.nn.functional as F

    if after.shape != before.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    H, W = before.shape[:2]
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    prep = lambda im: F.pad(to_tensor(im)[None], (0, pw, 0, ph), mode="reflect").to(device)

    with torch.no_grad():
        a, b = prep(before), prep(after)
        flips = [[]] + ([[-1], [-2], [-1, -2]] if tta else [])
        acc = 0
        for d in flips:
            aa = a.flip(d) if d else a
            bb = b.flip(d) if d else b
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                p = torch.sigmoid(model(aa, bb)).float()
            acc = acc + (p.flip(d) if d else p)
        return (acc / len(flips))[0, 0, :H, :W].cpu().numpy()


def clean_mask(prob, thr, min_area=MIN_AREA_PX):
    """Threshold -> morphological opening -> drop tiny blobs."""
    m = (prob > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


# ---------------- 2. Protected-zone extraction (before vs before_zone) ----------------
def extract_zone(before, zone_img, thr=15, min_area=500):
    """The red highlight raises (R-G) and (R-B) relative to the original image."""
    if zone_img.shape != before.shape:
        zone_img = cv2.resize(zone_img, (before.shape[1], before.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    b, z = before.astype(np.int16), zone_img.astype(np.int16)
    d_rg = (z[..., 0] - z[..., 1]) - (b[..., 0] - b[..., 1])
    d_rb = (z[..., 0] - z[..., 2]) - (b[..., 0] - b[..., 2])
    m = ((d_rg > thr) & (d_rb > thr)).astype(np.uint8)
    m = cv2.medianBlur(m * 255, 5) > 127
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for c in cnts:  # fill holes, drop specks
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 1, thickness=-1)
    return out


# ---------------- 3. Mask -> Shapely geometry ----------------
def mask_to_polygon(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        pts = c[:, 0, :]
        p = Polygon(pts).buffer(0) if len(pts) >= 3 else None
        if p is None or p.is_empty or p.area == 0:
            x, y, w, h = cv2.boundingRect(c)
            p = box(x, y, x + w, y + h)
        polys.append(p)
    return unary_union(polys) if polys else Polygon()


# ---------------- 4. Full analysis of one pair ----------------
def analyze(prob, before, zone_img, thr):
    """Same logic as notebook analyze(), but takes the probability map
    (so the model call can be timed / swapped independently)."""
    change = clean_mask(prob, thr)
    zone = extract_zone(before, zone_img) if zone_img is not None else np.zeros(change.shape, np.uint8)
    zone_poly = mask_to_polygon(zone)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(change, connectivity=8)
    rows, features = [], []
    for i in range(1, n):
        comp = (labels == i)
        poly = mask_to_polygon(comp)
        px_tot, px_in = int(comp.sum()), int((comp & (zone > 0)).sum())
        hit = (not zone_poly.is_empty) and poly.intersects(zone_poly) and px_in >= MIN_OVERLAP_PX
        pct = 100 * px_in / px_tot
        status = "COMPLIANT" if not hit else ("ENCROACHMENT (major)" if pct >= 50 else "ENCROACHMENT (partial)")
        row = dict(id=i, cx=round(float(cents[i][0]), 1), cy=round(float(cents[i][1]), 1),
                   area_m2=round(px_tot * PIX_AREA_M2, 1),
                   inside_m2=round(px_in * PIX_AREA_M2, 1),
                   inside_pct=round(pct, 1), status=status, flag=bool(hit))
        rows.append(row)
        features.append(dict(type="Feature", properties=row, geometry=mapping(poly)))

    flagged = [r for r in rows if r["flag"]]
    summary = dict(n_objects=len(rows),
                   total_m2=float(change.sum() * PIX_AREA_M2),
                   n_flagged=len(flagged),
                   breach_m2=float(sum(r["inside_m2"] for r in flagged)),
                   zone_pct=float(100 * zone.mean()),
                   has_zone=zone_img is not None)
    geojson = dict(type="FeatureCollection",
                   properties=dict(crs="pixel coordinates (no CRS)", pixel_size_m=PIXEL_SIZE_M),
                   features=features)
    return dict(prob=prob, change=change, zone=zone, labels=labels,
                rows=rows, summary=summary, geojson=geojson)


# ---------------- 5. Rendering the two output images ----------------
COL = dict(change=(255, 215, 0), ok=(0, 220, 0), bad=(255, 0, 0), zone=(255, 140, 0))


def tint(img, mask, color, alpha):
    out, m = img.copy(), mask.astype(bool)
    out[m] = (img[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def outline(img, mask, color, th=2):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, th)
    return img


def banner(img, text, color):
    band = np.zeros((46, img.shape[1], 3), np.uint8)
    cv2.putText(band, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return np.vstack([band, img])


def render_change(after, res):
    img = tint(after, res["change"], COL["change"], 0.45)
    img = outline(img, res["change"], COL["change"], 2)
    s = res["summary"]
    return banner(img, f"Detected new construction: {s['n_objects']} objects, {s['total_m2']:.0f} m2",
                  COL["change"])


def render_compliance(after, res):
    img = tint(after, res["zone"], COL["zone"], 0.25)
    img = outline(img, res["zone"], COL["zone"], 3)
    for r in res["rows"]:
        m = res["labels"] == r["id"]
        c = COL["bad"] if r["flag"] else COL["ok"]
        img = outline(tint(img, m, c, 0.55), m, c, 2)
    for r in res["rows"]:
        if not r["flag"]:
            continue
        pos = (int(r["cx"]) + 8, int(r["cy"]) - 8)
        cv2.putText(img, f"#{r['id']}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"#{r['id']}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    s = res["summary"]
    if s["n_flagged"]:
        return banner(img, f"ALERT: {s['n_flagged']} encroachment(s), {s['breach_m2']:.0f} m2 inside protected zone",
                      (255, 60, 60))
    return banner(img, "No encroachment detected (orange = zone, green = compliant)", (0, 220, 0))


# ---------------- Extra: transparent overlays for the web viewer ----------------
def _paint(out, mask, color, alpha):
    m = mask.astype(np.uint8)
    out[m.astype(bool)] = (*color, alpha)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = np.zeros(m.shape, np.uint8)
    cv2.drawContours(edge, cnts, -1, 1, 2)
    out[edge.astype(bool)] = (*color, 255)


def zone_overlay(zone):
    """Transparent PNG layer of the protected zone (orange)."""
    out = np.zeros((*zone.shape, 4), np.uint8)
    _paint(out, zone, COL["zone"], 70)
    return out


def change_overlay(res):
    """Transparent PNG layer of detected objects: red = encroaching, green = compliant
    (yellow for everything when no zone was supplied)."""
    out = np.zeros((*res["change"].shape, 4), np.uint8)
    for r in res["rows"]:
        m = res["labels"] == r["id"]
        if res["summary"]["has_zone"]:
            c = COL["bad"] if r["flag"] else COL["ok"]
        else:
            c = COL["change"]
        _paint(out, m, c, 120)
    return out
