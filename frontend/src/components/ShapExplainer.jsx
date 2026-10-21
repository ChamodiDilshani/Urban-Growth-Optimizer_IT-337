import React, { useState } from 'react';

export default function ShapExplainer({ shapImage, isLoading }) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleDownload = () => {
    if (!shapImage) return;
    const link = document.createElement('a');
    link.href = shapImage;
    link.download = `SHAP_Price_Justification_${new Date().toISOString().slice(0, 10)}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div style={{ marginTop: '1rem' }}>
      {/* Section Header & Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <h2 className="section-title" style={{ fontSize: '1.35rem', margin: 0 }}>
              📊 3. Explainable AI (SHAP) Price Justification & Feature Attribution
            </h2>
            <span style={{
              backgroundColor: '#eff6ff',
              color: '#1d4ed8',
              fontSize: '0.75rem',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: '12px',
              border: '1px solid #bfdbfe'
            }}>
              Full-Spectrum Attribution
            </span>
          </div>
          <p className="section-caption" style={{ marginTop: '0.35rem', maxWidth: '1000px', fontSize: '0.9rem' }}>
            Visualizing local feature contributions (waterfall plot) that push the base expected valuation to the final predicted land value. Each bar quantifies how a specific geospatial, municipal, or physical variable increases or decreases the estimated price per perch (LKR).
          </p>

          {/* Legend Guide */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.6rem', alignItems: 'center' }}>
            <div style={{
              fontSize: '0.78rem',
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: '#fee2e2',
              color: '#991b1b',
              border: '1px solid #fecaca',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#ef4444' }}></span>
              <span>Red Bar (+): Increases Parcel Valuation</span>
            </div>

            <div style={{
              fontSize: '0.78rem',
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: '#e0f2fe',
              color: '#075985',
              border: '1px solid #bae6fd',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#0284c7' }}></span>
              <span>Blue Bar (-): Decreases Parcel Valuation</span>
            </div>

            <div style={{
              fontSize: '0.78rem',
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: '#f1f5f9',
              color: '#334155',
              border: '1px solid #cbd5e1',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
              <span>📌 E[f(x)]: District Base Expectation</span>
            </div>

            <div style={{
              fontSize: '0.78rem',
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: '#f0fdf4',
              color: '#166534',
              border: '1px solid #bbf7d0',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
              <span>🎯 f(x): Predicted Parcel Value</span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        {shapImage && !isLoading && (
          <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.25rem' }}>
            <button
              type="button"
              onClick={() => setIsFullscreen(true)}
              title="View chart in fullscreen popup"
              style={{
                fontSize: '0.8rem',
                padding: '6px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                backgroundColor: '#ffffff',
                color: '#334155',
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem'
              }}
            >
              <span>🔍</span> Full View
            </button>
            <button
              type="button"
              onClick={handleDownload}
              title="Download high-resolution SHAP chart"
              style={{
                fontSize: '0.8rem',
                padding: '6px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                backgroundColor: '#ffffff',
                color: '#334155',
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem'
              }}
            >
              <span>📥</span> Download PNG
            </button>
          </div>
        )}
      </div>

      {/* Main Full-Width SHAP Canvas Card */}
      <div className="shap-card">
        {isLoading ? (
          <div className="shap-placeholder">
            <div style={{ fontSize: '1.5rem', animation: 'spin 1s linear infinite' }}>⏳</div>
            <strong style={{ color: 'var(--text-primary)' }}>Computing TreeExplainer SHAP Values...</strong>
            <span style={{ fontSize: '0.85rem' }}>Calculating exact local gradient feature attributions for this parcel.</span>
          </div>
        ) : shapImage ? (
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <img
              src={shapImage}
              alt="SHAP Waterfall Price Justification"
              className="shap-image"
              style={{ maxWidth: '6.8in', maxHeight: '4.5in', width: 'auto', height: 'auto' }}
            />
          </div>
        ) : (
          <div className="shap-placeholder">
            <span>No SHAP explanation available yet. Select parcel parameters to run inference.</span>
          </div>
        )}
      </div>

      {/* Fullscreen Modal / Lightbox */}
      {isFullscreen && shapImage && (
        <div
          onClick={() => setIsFullscreen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.85)',
            backdropFilter: 'blur(4px)',
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              padding: '1.5rem',
              maxWidth: '96vw',
              maxHeight: '92vh',
              overflow: 'auto',
              boxShadow: '0 20px 40px rgba(0,0,0,0.4)',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#0f172a' }}>
                SHAP Waterfall Feature Attribution (High-Resolution View)
              </h3>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  type="button"
                  onClick={handleDownload}
                  style={{
                    padding: '4px 10px',
                    borderRadius: '4px',
                    border: '1px solid #cbd5e1',
                    backgroundColor: '#f8fafc',
                    cursor: 'pointer',
                    fontSize: '0.82rem',
                    fontWeight: 600,
                  }}
                >
                  📥 Download
                </button>
                <button
                  type="button"
                  onClick={() => setIsFullscreen(false)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: '4px',
                    border: '1px solid #cbd5e1',
                    backgroundColor: '#f1f5f9',
                    cursor: 'pointer',
                    fontSize: '0.85rem',
                    fontWeight: 700,
                  }}
                >
                  ✕ Close
                </button>
              </div>
            </div>

            <img
              src={shapImage}
              alt="SHAP Waterfall High Res"
              style={{
                width: '100%',
                maxHeight: '78vh',
                objectFit: 'contain',
                borderRadius: '6px',
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
