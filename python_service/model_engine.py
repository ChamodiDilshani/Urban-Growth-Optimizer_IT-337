import os
import sys
import math
import io
import base64
import threading
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

plot_lock = threading.Lock()
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
import sklearn.compose._column_transformer

# Backwards compatibility patch for scikit-learn unpickling colombo_data_preprocessor.pkl
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "colombo_land_valuation_xgb.json")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "colombo_data_preprocessor.pkl")
EXCEL_PATH = os.path.join(BASE_DIR, "Land prices of Colombo district-Sri Lanka.xlsx")

# Flood prone regions
FLOOD_PRONE = [
    'kaduwela', 'ranala', 'bomiriya', 'kolonnawa', 'wellampitiya',
    'kotikawatta', 'angoda', 'kelanimulla', 'ambathale', 'hanwella'
]

ROAD_MULTIPLIERS = {
    "10 ft": 0.92,
    "12 ft": 0.96,
    "15 ft": 1.00,
    "20 ft": 1.06,
    "30+ ft": 1.12
}

FRIENDLY_NAMES = {
    'Distance from fort': 'Distance to Fort',
    'Land_size(Perches)': 'Land Size (Perches)',
    'count_banks_within_2km': 'Banks within 2km',
    'min_dist_nearest_Pvt_Hospital': 'Dist to Pvt Hospital',
    'min_dist_govtschools_a': 'Dist to Govt School (A)',
    'min_dist_govtschools_b': 'Dist to Govt School (B)',
    'min_dist_semigovtschools': 'Dist to Semi-Govt School',
    'min_dist_intlschools': 'Dist to Intl School',
    'min_dist_uni': 'Dist to University',
    'min_dist_nearest_railway': 'Dist to Railway Station',
    'min_dist_nearest_bank': 'Dist to Nearest Bank',
    'min_dist_nearest_FinanceCompany': 'Dist to Finance Co',
    'count_FinanceCompanies_within_2km': 'Finance Co within 2km',
    'min_dist_nearest_Govt_Hospital': 'Dist to Govt Hospital',
    'count_Govt_Hospitals': 'Govt Hospitals within 2km',
    'count_Pvt_Hospital': 'Pvt Hospitals within 2km',
    'min_dist_nearest_Pvt_Med_center': 'Dist to Pvt Clinic',
    'count_Pvt_Med_Centers': 'Pvt Clinics within 2km',
    'min_dist_nearest_Supermarket': 'Dist to Supermarket',
    'count_Supermarkets_within2km': 'Supermarkets within 2km',
    'min_dist_nearest_Fuel_station': 'Dist to Fuel Station',
    'count_Fuel_Stations_within2km': 'Fuel Stations within 2km',
    'min_dist_nearest_express': 'Dist to Express Highway',
}

def get_friendly_name(raw_name: str) -> str:
    if raw_name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[raw_name]
    if raw_name.startswith('Address_'):
        sub = raw_name.replace('Address_', '').replace('_', ' ').title()
        return f"Location = {sub}"
    if raw_name.startswith('Land_type_'):
        lt = raw_name.replace('Land_type_', '').replace('_', ' ').title()
        return f"Zoning = {lt}"
    return raw_name.replace('_', ' ').title()

def render_publication_waterfall(shap_vals, input_dict: dict, max_display: int = 8, figsize=(6.8, 4.4), dpi: int = 180) -> str:
    """Renders a clean, high-fidelity SHAP waterfall plot matching standard publication styling."""
    base_val = float(shap_vals.base_values[0])
    values = shap_vals.values[0]
    raw_feature_names = list(shap_vals.feature_names)
    raw_data = shap_vals.data[0]

    # Sort indices by absolute SHAP contribution
    sorted_idx = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    top_idx = sorted_idx[:max_display]
    other_idx = sorted_idx[max_display:]

    display_items = []
    if other_idx:
        other_sum = sum(values[i] for i in other_idx)
        display_items.append({
            'name': f"{len(other_idx)} other features",
            'val_str': "",
            'delta': other_sum
        })

    for i in reversed(top_idx):
        raw_name = raw_feature_names[i]
        friendly_name = get_friendly_name(raw_name)

        orig_val = input_dict.get(raw_name, None)
        if orig_val is not None:
            if 'dist' in raw_name.lower():
                val_str = f"{float(orig_val):.1f}km"
            elif 'size' in raw_name.lower():
                val_str = f"{float(orig_val):.0f}p"
            elif isinstance(orig_val, (int, float)):
                if abs(float(orig_val) - round(float(orig_val))) < 0.001:
                    val_str = str(int(round(float(orig_val))))
                else:
                    val_str = f"{float(orig_val):.2g}"
            else:
                val_str = str(orig_val)
        elif raw_name.startswith('Address_') or raw_name.startswith('Land_type_'):
            val_str = "1" if raw_data[i] > 0.5 else "0"
        else:
            scale_val = raw_data[i]
            if abs(scale_val - round(scale_val)) < 0.001:
                val_str = str(int(round(scale_val)))
            else:
                val_str = f"{scale_val:.2g}"

        display_items.append({
            'name': friendly_name,
            'val_str': val_str,
            'delta': float(values[i])
        })

    # Plot
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')

    COLOR_POS = "#ff0051"  # Vibrant SHAP Red / Pink
    COLOR_NEG = "#008bfb"  # Vibrant SHAP Blue
    COLOR_TEXT = "#1e293b"
    COLOR_MUTED = "#64748b"
    COLOR_GRID = "#f8fafc"
    COLOR_LINE = "#94a3b8"

    n = len(display_items)
    y_positions = np.arange(n)
    bar_height = 0.55

    cum_x = [base_val]
    for item in display_items:
        cum_x.append(cum_x[-1] + item['delta'])

    final_val = cum_x[-1]
    all_x = cum_x.copy()
    min_x, max_x = min(all_x), max(all_x)
    x_range = max(1.0, max_x - min_x)
    arrow_w = x_range * 0.024

    # Subtle horizontal guidelines
    for y in y_positions:
        ax.axhline(y, color=COLOR_GRID, linestyle=':', linewidth=1.1, zorder=0)

    # Vertical reference lines
    ax.axvline(base_val, color=COLOR_LINE, linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(final_val, color=COLOR_LINE, linestyle="--", linewidth=1.1, zorder=1)

    # Draw bars
    for i in range(n):
        y = y_positions[i]
        x_start = cum_x[i]
        x_end = cum_x[i+1]
        delta = display_items[i]['delta']
        is_pos = delta >= 0
        color = COLOR_POS if is_pos else COLOR_NEG

        w = abs(delta)
        y_bot = y - bar_height / 2
        y_top = y + bar_height / 2

        if is_pos:
            aw = min(arrow_w, w * 0.42)
            pts = [
                [x_start, y_bot],
                [x_end - aw, y_bot],
                [x_end, y],
                [x_end - aw, y_top],
                [x_start, y_top]
            ]
        else:
            aw = min(arrow_w, w * 0.42)
            pts = [
                [x_start, y_bot],
                [x_end + aw, y_bot],
                [x_end, y],
                [x_end + aw, y_top],
                [x_start, y_top]
            ]

        poly = patches.Polygon(pts, closed=True, facecolor=color, edgecolor='none', zorder=3)
        ax.add_patch(poly)

        # Delta label
        sign = "+" if is_pos else "-"
        abs_d = abs(delta)
        if abs_d >= 1e6:
            val_text = f"{sign}{abs_d/1e6:.2f}M"
        elif abs_d >= 1e3:
            val_text = f"{sign}{abs_d/1e3:.1f}k"
        else:
            val_text = f"{sign}{abs_d:,.0f}"

        bar_frac = w / x_range
        if bar_frac > 0.17:
            text_x = (x_start + x_end) / 2
            ax.text(text_x, y, val_text, color="white", fontsize=8.6, fontweight="bold",
                    ha="center", va="center", zorder=4)
        else:
            if is_pos:
                text_x = x_end + x_range * 0.016
                ha = "left"
            else:
                text_x = x_end - x_range * 0.016
                ha = "right"
            ax.text(text_x, y, val_text, color=color, fontsize=8.6, fontweight="bold",
                    ha=ha, va="center", zorder=4)

    # Y-axis labels
    ax.set_yticks(y_positions)
    y_labels = []
    for item in display_items:
        if item['val_str']:
            y_labels.append(f"{item['val_str']} = {item['name']}")
        else:
            y_labels.append(item['name'])

    ax.set_yticklabels(y_labels, fontsize=8.8, color=COLOR_TEXT)
    ax.tick_params(axis='y', length=0, pad=10)

    # X axis padding & limits
    pad_left = x_range * 0.14
    pad_right = x_range * 0.14
    ax.set_xlim(min_x - pad_left, max_x + pad_right)
    ax.set_ylim(-0.95, n + 0.35)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['bottom'].set_linewidth(1.1)

    ticks = ax.get_xticks()
    new_ticks = [t for t in ticks if (min_x - pad_left * 0.75) <= t <= (max_x + pad_right * 0.75)]
    if new_ticks:
        ax.set_xticks(new_ticks)
        tick_labels = []
        for t in new_ticks:
            if abs(t) >= 1e6:
                tick_labels.append(f"{t/1e6:.2f}")
            elif abs(t) >= 1e3:
                tick_labels.append(f"{t/1e3:.0f}k")
            else:
                tick_labels.append(f"{t:,.0f}")
        ax.set_xticklabels(tick_labels, fontsize=8.5, color='#334155')
    ax.tick_params(axis='x', colors='#334155', length=4, pad=4)

    # Bottom E[f(X)] text
    base_str = f"{base_val/1e6:.3f}" if abs(base_val) >= 1e6 else f"{base_val:,.0f}"
    ax.text(base_val, -1.55, f"$E[f(X)] = {base_str}$", color=COLOR_MUTED, fontsize=9.2,
            ha="center", va="top", fontweight="bold")

    # Top f(x) text
    final_str = f"{final_val/1e6:.3f}" if abs(final_val) >= 1e6 else f"{final_val:,.0f}"
    ax.text(final_val, n + 0.12, f"$f(x) = {final_str}$", color=COLOR_TEXT, fontsize=9.5,
            ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, facecolor='white')
    plt.close('all')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

class ModelEngine:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.df_ref = None
        self.explainer = None
        self.suburbs = []
        self.land_types = []
        self._load_artifacts()

    def _load_artifacts(self):
        print("Loading reference dataset from:", EXCEL_PATH)
        self.df_ref = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
        self.suburbs = sorted([str(s) for s in self.df_ref['Address'].dropna().unique()])
        self.land_types = sorted([str(t) for t in self.df_ref['Land_type'].dropna().unique()])

        print("Loading XGBoost model from:", MODEL_PATH)
        self.model = XGBRegressor()
        self.model.load_model(MODEL_PATH)

        # Load original trained preprocessor
        print("Loading trained preprocessor from:", PREPROCESSOR_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        # Test transform to verify preprocessor compatibility
        test_row = self.df_ref.iloc[:1].copy()
        drop_cols = ['Address_ID', 'Posted_Date_new', 'Mentioned Price(Rs)', 'Price_Scale', 'Price per Perch']
        X_test = test_row.drop(columns=drop_cols, errors='ignore')
        self.preprocessor.transform(X_test)
        print("Successfully loaded original trained preprocessor!")

        print("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("ModelEngine initialized successfully!")

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float = 6.9344, lon2: float = 79.8428) -> float:
        """Haversine distance in km between two GPS coordinates."""
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0)**2
        return round(2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)), 3)

    def is_flood_risk(self, suburb: str) -> bool:
        return suburb.lower().strip() in FLOOD_PRONE

    def predict(self, suburb: str, land_size: float, land_type: str, road_width: str,
                target_year: int, lat: float, lon: float) -> dict:
        dist_fort = self.haversine_distance(lat, lon, 6.9344, 79.8428)
        flood_risk = self.is_flood_risk(suburb)

        # Prepare single row for inference
        suburb_profile = self.df_ref[self.df_ref['Address'] == suburb].median(numeric_only=True).to_dict()
        drop_cols = ['Address_ID', 'Posted_Date_new', 'Mentioned Price(Rs)', 'Price_Scale', 'Price per Perch']
        input_dict = {col: suburb_profile.get(col, 0) for col in self.df_ref.columns if col not in drop_cols}
        input_dict['Address'] = suburb
        input_dict['Land_size(Perches)'] = land_size
        input_dict['Land_type'] = land_type
        input_dict['Distance from fort'] = dist_fort

        input_df = pd.DataFrame([input_dict])
        X_transformed = self.preprocessor.transform(input_df)

        # Model base prediction
        base_pred_per_perch = float(self.model.predict(X_transformed)[0])

        # Multipliers
        road_mult = ROAD_MULTIPLIERS.get(road_width, 1.0)
        hazard_mult = 0.90 if flood_risk else 1.00
        years_ahead = max(0, target_year - 2026)
        appreciation_mult = (1.06) ** years_ahead

        final_price_per_perch = round(base_pred_per_perch * road_mult * hazard_mult * appreciation_mult, 2)
        total_valuation = round(final_price_per_perch * land_size, 2)

        # Statutory Tax Calculation
        yield_rate = 0.06 if "Commercial" in land_type else (0.04 if "Residential" in land_type else 0.02)
        tax_rate = 0.15 if "Commercial" in land_type else (0.08 if "Residential" in land_type else 0.04)
        annual_assessed_value = round(total_valuation * yield_rate, 2)
        annual_tax = round(annual_assessed_value * tax_rate, 2)

        # 5-Year Horizon
        multi_year_tax = {
            f"Year {target_year + i}": round(annual_tax * ((1.05) ** i), 2)
            for i in range(5)
        }

        # SHAP Waterfall Generation
        cat_features = ['Address', 'Land_type']
        num_features = [c for c in input_df.columns if c not in cat_features]
        cat_transformer = self.preprocessor.named_transformers_['cat']
        feature_names = num_features + list(cat_transformer.get_feature_names_out(cat_features))

        X_trans_df = pd.DataFrame(X_transformed, columns=feature_names)
        shap_vals = self.explainer(X_trans_df)

        # Thread-safe publication-quality SHAP waterfall plot generation
        with plot_lock:
            try:
                shap_b64 = render_publication_waterfall(
                    shap_vals=shap_vals,
                    input_dict=input_dict,
                    max_display=8,
                    figsize=(6.8, 4.4),
                    dpi=180
                )
            except Exception as plot_err:
                print("SHAP plot warning:", plot_err)
                shap_b64 = None

        # Generate Official Assessment Tax PDF statement in Base64 (Bypasses IDM network interception)
        from pdf_generator import create_pdf_bytes
        parcel_info_pdf = {
            "Suburb / Area": suburb.title(),
            "Coordinates": f"{lat:.4f}, {lon:.4f}",
            "Land Extent": f"{land_size} Perches",
            "Zoning Category": land_type,
            "Access Road Width": road_width,
            "Distance to Colombo Fort": f"{dist_fort} km",
            "NBRO Hazard Status": "High Flood Risk" if flood_risk else "Low Risk Zone"
        }
        val_info_pdf = {
            "Predicted Price per Perch": f"LKR {final_price_per_perch:,.2f}",
            "Total Parcel Valuation": f"LKR {total_valuation:,.2f}",
            "Annual Assessed Rental Value": f"LKR {annual_assessed_value:,.2f}",
            f"Statutory Tax ({target_year})": f"LKR {annual_tax:,.2f}",
            "Multi-Year": multi_year_tax
        }
        pdf_bytes = create_pdf_bytes(parcel_info_pdf, val_info_pdf)
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_filename = f"Assessment_Tax_{suburb}_{target_year}.pdf"

        return {
            "suburb": suburb,
            "land_size": land_size,
            "land_type": land_type,
            "road_width": road_width,
            "target_year": target_year,
            "coordinates": {"lat": lat, "lon": lon},
            "dist_fort": dist_fort,
            "is_flood_risk": flood_risk,
            "base_pred_per_perch": base_pred_per_perch,
            "road_multiplier": road_mult,
            "hazard_multiplier": hazard_mult,
            "appreciation_multiplier": round(appreciation_mult, 4),
            "final_price_per_perch": final_price_per_perch,
            "total_valuation": total_valuation,
            "yield_rate": yield_rate,
            "tax_rate": tax_rate,
            "annual_assessed_value": annual_assessed_value,
            "annual_tax": annual_tax,
            "multi_year_tax": multi_year_tax,
            "shap_image": shap_b64,
            "pdf_base64": pdf_b64,
            "pdf_filename": pdf_filename
        }

# Global singleton
engine = ModelEngine()
