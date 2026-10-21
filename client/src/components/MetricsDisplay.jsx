import React, { useState } from 'react';

export default function MetricsDisplay({
  valuation,
  targetYear,
  onDownloadPdf,
  isDownloadingPdf,
}) {
  const [showMultiYear, setShowMultiYear] = useState(true);

  if (!valuation) {
    return (
      <div className="column-panel">
        <h2 className="section-title">💰 2. AI Market Valuation & Tax Forecast ({targetYear})</h2>
        <div className="st-alert st-alert-info">
          Enter parameters on the left to calculate property valuation and statutory tax liability.
        </div>
      </div>
    );
  }

  const {
    final_price_per_perch = 0,
    total_valuation = 0,
    annual_assessed_value = 0,
    annual_tax = 0,
    tax_rate = 0.08,
    multi_year_tax = {},
  } = valuation;

  const formatLKR = (val) => {
    return 'LKR ' + Number(val || 0).toLocaleString('en-US', {
      maximumFractionDigits: 0,
    });
  };

  return (
    <div className="column-panel">
      <h2 className="section-title">
        💰 2. AI Market Valuation & Tax Forecast ({targetYear})
      </h2>

      {/* 2x2 Metric Cards Grid */}
      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-label">Predicted Value per Perch</span>
          <span className="metric-value">{formatLKR(final_price_per_perch)}</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Total Parcel Valuation</span>
          <span className="metric-value">{formatLKR(total_valuation)}</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Annual Assessed Value</span>
          <span className="metric-value">{formatLKR(annual_assessed_value)}</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Statutory Annual Tax (වරිපනම්)</span>
          <span className="metric-value">{formatLKR(annual_tax)}</span>
          <span className="metric-delta">
            {(tax_rate * 100).toFixed(0)}% Municipal Rate
          </span>
        </div>
      </div>

      {/* Official PDF Download Button (Direct in-memory Base64 download to bypass IDM interception) */}
      {valuation.pdf_base64 ? (
        <a
          href={`data:application/pdf;base64,${valuation.pdf_base64}`}
          download={valuation.pdf_filename || `Assessment_Tax_Statement_${targetYear}.pdf`}
          className="pdf-btn"
          style={{ textDecoration: 'none' }}
        >
          <span>📄</span>
          <span>Download Official Assessment Tax Statement (PDF)</span>
        </a>
      ) : (
        <button
          type="button"
          className="pdf-btn"
          onClick={onDownloadPdf}
          disabled={isDownloadingPdf}
        >
          <span>📄</span>
          <span>
            {isDownloadingPdf
              ? 'Generating Official Statement...'
              : 'Download Official Assessment Tax Statement (PDF)'}
          </span>
        </button>
      )}

      {/* Multi-Year Tax Horizon Accordion/Table */}
      <div className="multi-year-section">
        <div
          className="multi-year-header"
          onClick={() => setShowMultiYear(!showMultiYear)}
          style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
        >
          <span>🗓️ 5-Year Statutory Tax Liability Forecast</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {showMultiYear ? '▲ Hide' : '▼ View Forecast'}
          </span>
        </div>

        {showMultiYear && (
          <table className="tax-table">
            <thead>
              <tr>
                <th>Projection Year</th>
                <th>Projected Annual Tax (LKR)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(multi_year_tax).map(([yearLabel, amount]) => (
                <tr key={yearLabel}>
                  <td style={{ fontWeight: 600 }}>{yearLabel}</td>
                  <td>
                    LKR{' '}
                    {Number(amount).toLocaleString('en-US', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
