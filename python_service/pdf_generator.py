from fpdf import FPDF
from datetime import datetime

def create_pdf_bytes(parcel_info: dict, valuation_info: dict) -> bytes:
    """
    Generates the official Predictive Assessment Tax Statement (WARIPANAM) PDF.
    Preserves exact format from original Streamlit app.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "URBANSCAPES-LK: AUTOMATED PROPERTY VALUATION ENGINE", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "OFFICIAL PREDICTIVE ASSESSMENT TAX STATEMENT (WARIPANAM)", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | District: Colombo", ln=True, align="C")
    pdf.ln(6)
    
    # Section 1
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. Property & Spatial Parcel Profile", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in parcel_info.items():
        pdf.cell(85, 7, f"   - {k}:", border=0)
        pdf.cell(0, 7, f"{v}", ln=True)
    pdf.ln(4)
    
    # Section 2
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. AI Market Valuation & Statutory Tax Calculation", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in valuation_info.items():
        if k != "Multi-Year":
            pdf.cell(85, 7, f"   - {k}:", border=0)
            pdf.cell(0, 7, f"{v}", ln=True)
    pdf.ln(4)
    
    # Section 3
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. Multi-Year Statutory Tax Liability Forecast (5-Year Horizon)", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Projection Year", border=1, align="C")
    pdf.cell(80, 8, "Projected Annual Tax (LKR)", border=1, align="C", ln=True)
    pdf.set_font("Helvetica", "", 10)
    multi_year = valuation_info.get("Multi-Year", {})
    for yr, amt in multi_year.items():
        pdf.cell(60, 7, str(yr), border=1, align="C")
        amt_str = f"LKR {amt:,.2f}" if isinstance(amt, (int, float)) else str(amt)
        pdf.cell(80, 7, amt_str, border=1, align="C", ln=True)
        
    return bytes(pdf.output())
