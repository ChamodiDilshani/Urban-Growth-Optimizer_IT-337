import React from 'react';

export default function ShapExplainer({ shapImage, isLoading }) {
  return (
    <div className="column-panel" style={{ marginTop: '0.5rem' }}>
      <h2 className="section-title">📊 3. Explainable AI (SHAP) Price Justification</h2>
      <p className="section-caption">
        Visualizing local feature contributions (waterfall plot) that push the base expected valuation to the final predicted land value.
      </p>

      <div className="shap-card">
        {isLoading ? (
          <div className="shap-placeholder">
            <span>⏳ Generating SHAP TreeExplainer feature attributions...</span>
          </div>
        ) : shapImage ? (
          <img
            src={shapImage}
            alt="SHAP Waterfall Price Justification"
            className="shap-image"
          />
        ) : (
          <div className="shap-placeholder">
            <span>No SHAP explanation available yet.</span>
          </div>
        )}
      </div>
    </div>
  );
}
