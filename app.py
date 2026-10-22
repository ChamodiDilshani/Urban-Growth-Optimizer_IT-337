import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import math
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Urbanscapes-LK | Property Valuation & Tax Engine", layout="wide")

# 1. Load Trained Model (.json), Preprocessor, and Reference Data
@st.cache_resource
def load_artifacts():
    df_ref = pd.read_excel("Land prices of Colombo district-Sri Lanka.xlsx", sheet_name="Sheet1")
    
    # Creating a new XGBoost model and loading a .json file
    model = XGBRegressor()
    model.load_model("colombo_land_valuation_xgb.json")
    
    # Preprocessor Load 
    try:
        preprocessor = joblib.load("colombo_data_preprocessor.pkl")
    except Exception:
        df_sorted = df_ref.sort_values('Posted_Date_new').reset_index(drop=True)
        Q1 = df_sorted['Price per Perch'].quantile(0.25)
        Q3 = df_sorted['Price per Perch'].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_sorted[(df_sorted['Price per Perch'] >= Q1 - 1.5 * IQR) & (df_sorted['Price per Perch'] <= Q3 + 1.5 * IQR)].copy()
        drop_cols = ['Address_ID', 'Posted_Date_new', 'Mentioned Price(Rs)', 'Price_Scale', 'Price per Perch']
        X = df_clean.drop(columns=drop_cols)
        cat_features = ['Address', 'Land_type']
        num_features = [col for col in X.columns if col not in cat_features]
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
            ]
        )
        preprocessor.fit(X.iloc[:int(len(df_clean) * 0.70)])
        
    return model, preprocessor, df_ref

model, preprocessor, df_ref = load_artifacts()

# Haversine Formula
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0)**2
    return round(2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)), 3)

def is_point_in_polygon(lat, lon, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > lon) != (yj > lon)) and (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
    return inside

NBRO_FLOOD_ZONES = [
    {
        'name': 'Malabe (SLIIT - Kothalawala Wetland Corridor)',
        'center': (6.9145, 79.9728),
        'radius_km': 1.35,
        'polygon': [
            (6.9240, 79.9680), (6.9260, 79.9770), (6.9190, 79.9830),
            (6.9100, 79.9800), (6.9060, 79.9710), (6.9080, 79.9640), (6.9170, 79.9650)
        ]
    },
    {
        'name': 'Kaduwela - Nawagamuwa (Kelani River Basin)',
        'center': (6.9330, 79.9840),
        'radius_km': 1.80,
        'polygon': [
            (6.9450, 79.9700), (6.9480, 79.9950), (6.9280, 80.0100), (6.9200, 79.9850), (6.9250, 79.9680)
        ]
    },
    {
        'name': 'Kolonnawa - Wellampitiya - Sedawatta Basin',
        'center': (6.9400, 79.8880),
        'radius_km': 1.75,
        'polygon': [
            (6.9560, 79.8700), (6.9580, 79.8980), (6.9380, 79.9120), (6.9260, 79.8900), (6.9340, 79.8720)
        ]
    },
    {
        'name': 'Kotikawatta - Mulleriyawa - Angoda Basin',
        'center': (6.9320, 79.9180),
        'radius_km': 1.60,
        'polygon': [
            (6.9450, 79.9050), (6.9480, 79.9320), (6.9250, 79.9360), (6.9180, 79.9150)
        ]
    },
    {
        'name': 'Ambatale - Kelanimulla Corridor',
        'center': (6.9460, 79.9400),
        'radius_km': 1.40,
        'polygon': [
            (6.9550, 79.9280), (6.9580, 79.9550), (6.9380, 79.9560), (6.9350, 79.9300)
        ]
    },
    {
        'name': 'Bomiriya - Ranala - Hanwella Floodplain',
        'center': (6.9300, 80.0250),
        'radius_km': 2.20,
        'polygon': [
            (6.9450, 79.9980), (6.9480, 80.0550), (6.9150, 80.0600), (6.9120, 80.0100)
        ]
    },
    {
        'name': 'Diyawanna - Parliament Marsh Retention Basin',
        'center': (6.8960, 79.9130),
        'radius_km': 1.30,
        'polygon': [
            (6.9120, 79.9020), (6.9140, 79.9240), (6.8850, 79.9280), (6.8820, 79.9060)
        ]
    },
    {
        'name': 'Bellanwila - Attidiya Sanctuary Basin',
        'center': (6.8450, 79.8900),
        'radius_km': 1.25,
        'polygon': [
            (6.8580, 79.8800), (6.8600, 79.9020), (6.8320, 79.9040), (6.8300, 79.8820)
        ]
    },
    {
        'name': 'Bolgoda Lake Northern Basin (Dampe / Kesbewa)',
        'center': (6.8020, 79.9220),
        'radius_km': 1.20,
        'polygon': [
            (6.8150, 79.9100), (6.8160, 79.9350), (6.7900, 79.9380), (6.7880, 79.9120)
        ]
    }
]

FLOOD_PRONE = [
    'kaduwela', 'ranala', 'bomiriya', 'kolonnawa', 'wellampitiya',
    'kotikawatta', 'angoda', 'kelanimulla', 'ambathale', 'hanwella',
    'nawagamuwa', 'sedawatta', 'mulleriyawa', 'meetotamulla'
]

def check_flood_risk(lat, lon, suburb=''):
    for z in NBRO_FLOOD_ZONES:
        if is_point_in_polygon(lat, lon, z['polygon']):
            return True, z['name']
    for z in NBRO_FLOOD_ZONES:
        if haversine_distance(lat, lon, z['center'][0], z['center'][1]) <= z['radius_km']:
            return True, z['name']
    clean_sub = str(suburb).lower().strip()
    if clean_sub in FLOOD_PRONE:
        return True, f"{clean_sub.title()} (Kelani Basin Suburb)"
    return False, None

def compute_algorithmic_tax(total_valuation, land_size, land_type, dist_fort, flood_risk, suburb_profile, target_year=2026):
    fort_prox = max(0.0, min(1.0, 1.0 - (dist_fort / 35.0)))
    bank_cnt = float(suburb_profile.get('count_banks_within_2km', 10) or 10)
    bank_density = min(1.0, bank_cnt / 35.0)
    super_cnt = float(suburb_profile.get('count_Supermarkets_within2km', 5) or 5)
    super_density = min(1.0, super_cnt / 20.0)
    railway_dist = float(suburb_profile.get('min_dist_nearest_railway', 5.0) or 5.0)
    transit_factor = 1.0 / (1.0 + 0.12 * railway_dist)

    s_infra = (0.35 * fort_prox + 0.30 * bank_density + 0.20 * super_density + 0.15 * transit_factor)
    s_infra = max(0.10, min(0.98, s_infra))

    scale_elasticity = (12.0 / max(1.0, land_size)) ** 0.08

    lt = str(land_type).lower().strip()
    if 'commercial' in lt and 'residential' in lt:
        z_weight = 0.85
    elif 'commercial' in lt:
        z_weight = 1.00
    elif 'residential' in lt:
        z_weight = 0.65
    elif 'agricultural' in lt:
        z_weight = 0.35
    else:
        z_weight = 0.50

    y_base = 0.028
    yield_rate = y_base * (1.0 + 0.45 * s_infra + 0.40 * z_weight) * scale_elasticity
    yield_rate = max(0.025, min(0.075, yield_rate))

    annual_assessed_value = round(total_valuation * yield_rate, 2)

    arv_threshold = 600000.0
    tau_min = 0.055
    tau_max = 0.135
    norm_arv = annual_assessed_value / arv_threshold
    logistic_term = 1.0 / (1.0 + math.exp(-1.8 * (norm_arv - 1.0)))
    tax_rate = tau_min + (tau_max - tau_min) * logistic_term + 0.015 * s_infra
    tax_rate = max(0.05, min(0.15, tax_rate))

    annual_tax = round(annual_assessed_value * tax_rate, 2)

    multi_year_tax = {}
    cum_tax = annual_tax
    annual_growth_rates = {}
    for i in range(5):
        yr = target_year + i
        g = 0.042 + 0.028 * s_infra * math.exp(-0.15 * i) - (0.008 if flood_risk else 0.0)
        annual_growth_rates[f"Year {yr}"] = round(g * 100, 2)
        if i > 0:
            cum_tax *= (1.0 + g)
        multi_year_tax[f"Year {yr}"] = round(cum_tax, 2)

    return {
        "yield_rate": round(yield_rate, 4),
        "tax_rate": round(tax_rate, 4),
        "annual_assessed_value": annual_assessed_value,
        "annual_tax": annual_tax,
        "multi_year_tax": multi_year_tax,
        "annual_growth_rates": annual_growth_rates,
        "spatial_infrastructure_index": round(s_infra, 3)
    }

# PDF Report Generator Function
def create_pdf_bytes(parcel_info, valuation_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "URBANSCAPES-LK: AUTOMATED PROPERTY VALUATION ENGINE", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "OFFICIAL PREDICTIVE ASSESSMENT TAX STATEMENT (WARIPANAM)", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | District: Colombo", ln=True, align="C")
    pdf.ln(6)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. Property & Spatial Parcel Profile", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in parcel_info.items():
        pdf.cell(85, 7, f"   - {k}:", border=0)
        pdf.cell(0, 7, f"{v}", ln=True)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. AI Market Valuation & Statutory Tax Calculation", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in valuation_info.items():
        if k != "Multi-Year":
            pdf.cell(85, 7, f"   - {k}:", border=0)
            pdf.cell(0, 7, f"{v}", ln=True)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. Multi-Year Statutory Tax Liability Forecast (5-Year Horizon)", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Projection Year", border=1, align="C")
    pdf.cell(80, 8, "Projected Annual Tax (LKR)", border=1, align="C", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for yr, amt in valuation_info["Multi-Year"].items():
        pdf.cell(60, 7, str(yr), border=1, align="C")
        pdf.cell(80, 7, f"LKR {amt:,.2f}", border=1, align="C", ln=True)
        
    return bytes(pdf.output())

# Header
st.title("🏙️ Urbanscapes-LK: Automated Property Valuation & Assessment Tax Engine")
st.markdown("**Component 3:** AI-Driven Land Valuation, SHAP Explainability & Municipal Assessment Tax (වරිපනම් බදු) Forecasting")

# Sidebar Controls
st.sidebar.header("📋 Property Parcel Parameters")
selected_suburb = st.sidebar.selectbox("Select Suburb / Area (Colombo District)", sorted(df_ref['Address'].unique()))
land_size = st.sidebar.number_input("Land Extent (Perches)", min_value=1.0, max_value=200.0, value=12.0, step=0.5)
land_type = st.sidebar.selectbox("Statutory Zoning Category", sorted(df_ref['Land_type'].unique()))
road_width = st.sidebar.selectbox("Access Road Width (ft)", ["10 ft", "12 ft", "15 ft", "20 ft", "30+ ft"], index=2)
target_year = st.sidebar.slider("Target Projection Horizon (Year)", min_value=2026, max_value=2035, value=2026)

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("📍 1. Interactive GIS Parcel Pinning (Leaflet/OSM)")
    st.caption("Click anywhere on the map to pin the land parcel and compute geodesic distances via Haversine formula.")
    
    m = folium.Map(
        location=[6.8770, 79.9904],
        zoom_start=11,
        min_zoom=11,
        max_bounds=True,
        min_lat=6.712,
        max_lat=6.985,
        min_lon=79.818,
        max_lon=80.226
    )
    m.fit_bounds([[6.712, 79.818], [6.985, 80.226]])
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=380, width=650)
    
    if map_data and map_data.get("last_clicked"):
        pinned_lat = map_data["last_clicked"]["lat"]
        pinned_lon = map_data["last_clicked"]["lng"]
    else:
        pinned_lat, pinned_lon = 6.9022, 79.9172
        
    dist_fort = haversine_distance(pinned_lat, pinned_lon, 6.9344, 79.8428)
    st.info(f"**Pinned Coordinates:** Lat `{pinned_lat:.4f}`, Lon `{pinned_lon:.4f}` | **Distance to Fort:** `{dist_fort} km`")
    
    # NBRO Flood Hazard Evaluation based on spatial coordinates & suburb
    is_flood_risk, flood_zone_name = check_flood_risk(pinned_lat, pinned_lon, selected_suburb)
    
    if is_flood_risk:
        zone_label = f" ({flood_zone_name})" if flood_zone_name else ""
        st.error(f"⚠️ **NBRO Environmental Hazard Flag:** High Flood Vulnerability Zone Detected{zone_label} (-10% Risk Adjustment Applied)")
    else:
        st.success("✅ **NBRO Environmental Hazard Flag:** Low Risk Zone (Safe for Development)")

# Prepare Single Row for Model Inference
suburb_profile = df_ref[df_ref['Address'] == selected_suburb].median(numeric_only=True).to_dict()
drop_cols = ['Address_ID', 'Posted_Date_new', 'Mentioned Price(Rs)', 'Price_Scale', 'Price per Perch']
input_dict = {col: suburb_profile.get(col, 0) for col in df_ref.columns if col not in drop_cols}
input_dict['Address'] = selected_suburb
input_dict['Land_size(Perches)'] = land_size
input_dict['Land_type'] = land_type
input_dict['Distance from fort'] = dist_fort

input_df = pd.DataFrame([input_dict])
X_transformed = preprocessor.transform(input_df)

# Predict Market Value + Adjustments
base_pred_per_perch = float(model.predict(X_transformed)[0])
road_multiplier = {"10 ft": 0.92, "12 ft": 0.96, "15 ft": 1.00, "20 ft": 1.06, "30+ ft": 1.12}[road_width]
hazard_multiplier = 0.90 if is_flood_risk else 1.00
years_ahead = target_year - 2026
appreciation_multiplier = (1.06) ** years_ahead

final_price_per_perch = base_pred_per_perch * road_multiplier * hazard_multiplier * appreciation_multiplier
total_valuation = final_price_per_perch * land_size

# Algorithmic Statutory Assessment Tax Calculation (Hedonic Capitalization Model)
tax_calc = compute_algorithmic_tax(
    total_valuation=total_valuation,
    land_size=land_size,
    land_type=land_type,
    dist_fort=dist_fort,
    flood_risk=is_flood_risk,
    suburb_profile=suburb_profile,
    target_year=target_year
)
yield_rate = tax_calc["yield_rate"]
tax_rate = tax_calc["tax_rate"]
annual_assessed_value = tax_calc["annual_assessed_value"]
annual_tax = tax_calc["annual_tax"]
multi_yr_tax = tax_calc["multi_year_tax"]

with col2:
    st.subheader(f"💰 2. AI Market Valuation & Tax Forecast ({target_year})")
    m1, m2 = st.columns(2)
    m1.metric("Predicted Value per Perch", f"LKR {final_price_per_perch:,.0f}")
    m2.metric("Total Parcel Valuation", f"LKR {total_valuation:,.0f}")
    
    m3, m4 = st.columns(2)
    m3.metric("Annual Assessed Value", f"LKR {annual_assessed_value:,.0f}", delta=f"{yield_rate*100:.2f}% Yield")
    m4.metric("Statutory Annual Tax (වරිපනම්)", f"LKR {annual_tax:,.0f}", delta=f"{tax_rate*100:.2f}% Algorithmic Rate")
    
    # PDF Download Button
    parcel_info_pdf = {
        "Suburb / Area": selected_suburb.title(),
        "Coordinates": f"{pinned_lat:.4f}, {pinned_lon:.4f}",
        "Land Extent": f"{land_size} Perches",
        "Zoning Category": land_type,
        "Access Road Width": road_width,
        "Distance to Colombo Fort": f"{dist_fort} km",
        "NBRO Hazard Status": f"High Flood Risk ({flood_zone_name})" if is_flood_risk else "Low Risk Zone"
    }
    val_info_pdf = {
        "Predicted Price per Perch": f"LKR {final_price_per_perch:,.2f}",
        "Total Parcel Valuation": f"LKR {total_valuation:,.2f}",
        "Annual Assessed Value (ARV)": f"LKR {annual_assessed_value:,.2f}",
        "Assessment Methodology": "Algorithmic Hedonic Capitalization & Progressive Rating",
        "Effective Rental Yield Rate": f"{yield_rate*100:.2f}%",
        f"Statutory Tax ({target_year})": f"LKR {annual_tax:,.2f} ({tax_rate*100:.2f}% Algorithmic Rate)",
        "Multi-Year": multi_yr_tax
    }
    pdf_bytes = create_pdf_bytes(parcel_info_pdf, val_info_pdf)
    # PDF Download Button (Base64 Direct Download - IDM Popup එක මඟහරවා ගැනීමට)
    import base64

    multi_yr_tax = {f"Year {target_year + i}": round(annual_tax * ((1.05) ** i), 2) for i in range(5)}
    parcel_info_pdf = {
        "Suburb / Area": selected_suburb.title(),
        "Coordinates": f"{pinned_lat:.4f}, {pinned_lon:.4f}",
        "Land Extent": f"{land_size} Perches",
        "Zoning Category": land_type,
        "Access Road Width": road_width,
        "Distance to Colombo Fort": f"{dist_fort} km",
        "NBRO Hazard Status": "High Flood Risk" if is_flood_risk else "Low Risk Zone"
    }
    val_info_pdf = {
        "Predicted Price per Perch": f"LKR {final_price_per_perch:,.2f}",
        "Total Parcel Valuation": f"LKR {total_valuation:,.2f}",
        "Annual Assessed Rental Value": f"LKR {annual_assessed_value:,.2f}",
        f"Statutory Tax ({target_year})": f"LKR {annual_tax:,.2f}",
        "Multi-Year": multi_yr_tax
    }
    
    # PDF Bytes සකස් කර ගැනීම
    pdf_bytes = create_pdf_bytes(parcel_info_pdf, val_info_pdf)
    
    # Base64 Encoding මඟින් Direct Download Link එකක් සෑදීම
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    pdf_filename = f"Assessment_Tax_{selected_suburb}_{target_year}.pdf"
    
    href = f'''
        <a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}" style="
            text-decoration: none;
            background-color: #FF4B4B;
            color: white;
            padding: 9px 16px;
            border-radius: 6px;
            font-weight: 600;
            display: inline-block;
            text-align: center;
            width: 100%;
            margin-top: 10px;
            margin-bottom: 10px;
        ">
            📄 Download Official Assessment Tax Statement (PDF)
        </span>
    '''
    st.markdown(href, unsafe_allow_html=True)
    
    st.subheader("📊 3. Explainable AI (SHAP) Price Justification")
    cat_features = ['Address', 'Land_type']
    num_features = [c for c in input_df.columns if c not in cat_features]
    feature_names = num_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))
    
    X_trans_df = pd.DataFrame(X_transformed, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_trans_df)
    
    fig, ax = plt.subplots(figsize=(7, 3.8))
    shap.plots.waterfall(shap_vals[0], max_display=8, show=False)
    st.pyplot(fig)