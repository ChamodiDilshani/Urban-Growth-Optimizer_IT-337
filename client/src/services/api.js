const API_BASE = 'http://localhost:5000/api/valuation';

export const api = {
  // 1. Fetch metadata (suburbs list, zoning categories, road widths, coordinates)
  async getMetadata() {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) {
      throw new Error(`Failed to fetch metadata: ${res.statusText}`);
    }
    return res.json();
  },

  // 2. Predict market valuation + SHAP
  async predict(params) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.message || 'Prediction failed');
    }
    return res.json();
  },

  // 3. Download official PDF statement (Base64 Data URI to prevent IDM interception)
  async downloadPdf(parcelInfo, valuationInfo, filename = 'Assessment_Tax_Statement.pdf') {
    const res = await fetch(`${API_BASE}/pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parcel_info: parcelInfo, valuation_info: valuationInfo }),
    });

    if (!res.ok) {
      throw new Error('Failed to generate PDF');
    }

    const json = await res.json();
    if (!json.pdf_base64) {
      throw new Error('No PDF content returned');
    }

    // Direct in-memory base64 data download - does not trigger IDM network interception
    const link = document.createElement('a');
    link.href = `data:application/pdf;base64,${json.pdf_base64}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  },

  // 4. History API (MongoDB)
  async getHistory() {
    const res = await fetch(`${API_BASE}/history`);
    if (!res.ok) throw new Error('Failed to fetch history');
    return res.json();
  },

  async deleteHistoryItem(id) {
    const res = await fetch(`${API_BASE}/history/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete history record');
    return res.json();
  },

  async clearHistory() {
    const res = await fetch(`${API_BASE}/history`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to clear history');
    return res.json();
  },
};
