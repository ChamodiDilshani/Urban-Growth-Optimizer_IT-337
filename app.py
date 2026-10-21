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

# Flood Risk Check
flood_prone = ['kaduwela', 'ranala', 'bomiriya', 'kolonnawa', 'wellampitiya', 'kotikawatta', 'angoda', 'kelanimulla', 'ambathale', 'hanwella']
is_flood_risk = selected_suburb.lower().strip() in flood_prone

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
    
    if is_flood_risk:
        st.error("⚠️ **NBRO Environmental Hazard Flag:** High Flood Vulnerability Zone Detected (-10% Risk Adjustment Applied)")
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

# Statutory Assessment Tax Calculation
yield_rate = 0.06 if "Commercial" in land_type else (0.04 if "Residential" in land_type else 0.02)
tax_rate = 0.15 if "Commercial" in land_type else (0.08 if "Residential" in land_type else 0.04)
annual_assessed_value = total_valuation * yield_rate
annual_tax = annual_assessed_value * tax_rate

with col2:
    st.subheader(f"💰 2. AI Market Valuation & Tax Forecast ({target_year})")
    m1, m2 = st.columns(2)
    m1.metric("Predicted Value per Perch", f"LKR {final_price_per_perch:,.0f}")
    m2.metric("Total Parcel Valuation", f"LKR {total_valuation:,.0f}")
    
    m3, m4 = st.columns(2)
    m3.metric("Annual Assessed Value", f"LKR {annual_assessed_value:,.0f}")
    m4.metric("Statutory Annual Tax (වරිපනම්)", f"LKR {annual_tax:,.0f}", delta=f"{tax_rate*100:.0f}% Municipal Rate")
    
    # PDF Download Button
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