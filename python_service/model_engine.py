import os
import math
import io
import base64
import threading
import pandas as pd

plot_lock = threading.Lock()
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
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

        # Thread-safe SHAP waterfall plot generation
        with plot_lock:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            try:
                shap.plots.waterfall(shap_vals[0], max_display=8, show=False)
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=140)
                shap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception as plot_err:
                print("SHAP plot warning:", plot_err)
                shap_b64 = None
            finally:
                plt.close('all')

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
