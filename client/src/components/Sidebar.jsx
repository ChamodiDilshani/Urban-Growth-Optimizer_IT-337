import React from 'react';

export default function Sidebar({
  suburbs = [],
  landTypes = [],
  roadWidths = [],
  params,
  setParams,
  onOpenHistory,
  historyCount = 0
}) {
  const handleChange = (field, value) => {
    setParams((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <aside className="streamlit-sidebar">
      <h2 className="sidebar-title">
        <span>📋</span>
        <span>Property Parcel Parameters</span>
      </h2>

      {/* Suburb Selector */}
      <div className="form-group">
        <label className="form-label" htmlFor="suburb-select">
          Select Suburb / Area (Colombo District)
        </label>
        <select
          id="suburb-select"
          className="form-select"
          value={params.suburb}
          onChange={(e) => handleChange('suburb', e.target.value)}
        >
          {suburbs.map((suburb) => (
            <option key={suburb} value={suburb}>
              {suburb}
            </option>
          ))}
        </select>
      </div>

      {/* Land Extent */}
      <div className="form-group">
        <label className="form-label" htmlFor="land-size">
          Land Extent (Perches)
        </label>
        <input
          id="land-size"
          type="number"
          className="form-control"
          min="1.0"
          max="200.0"
          step="0.5"
          value={params.land_size}
          onChange={(e) => handleChange('land_size', parseFloat(e.target.value) || 1.0)}
        />
      </div>

      {/* Statutory Zoning Category */}
      <div className="form-group">
        <label className="form-label" htmlFor="land-type">
          Statutory Zoning Category
        </label>
        <select
          id="land-type"
          className="form-select"
          value={params.land_type}
          onChange={(e) => handleChange('land_type', e.target.value)}
        >
          {landTypes.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </div>

      {/* Access Road Width */}
      <div className="form-group">
        <label className="form-label" htmlFor="road-width">
          Access Road Width (ft)
        </label>
        <select
          id="road-width"
          className="form-select"
          value={params.road_width}
          onChange={(e) => handleChange('road_width', e.target.value)}
        >
          {roadWidths.map((w) => (
            <option key={w} value={w}>
              {w}
            </option>
          ))}
        </select>
      </div>

      {/* Target Year Horizon Slider */}
      <div className="form-group">
        <div className="range-slider-container">
          <label className="form-label" htmlFor="target-year">
            Target Projection Horizon (Year)
          </label>
          <div className="slider-val-display">
            <span>2026</span>
            <span style={{ color: 'var(--primary-color)', fontWeight: 700 }}>
              {params.target_year}
            </span>
            <span>2035</span>
          </div>
          <input
            id="target-year"
            type="range"
            min="2026"
            max="2035"
            step="1"
            value={params.target_year}
            onChange={(e) => handleChange('target_year', parseInt(e.target.value, 10))}
          />
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: '0.5rem 0' }} />

      {/* History Action */}
      <button
        type="button"
        onClick={onOpenHistory}
        style={{
          backgroundColor: '#ffffff',
          color: 'var(--text-primary)',
          border: '1px solid #d3d7df',
          padding: '8px 12px',
          borderRadius: '6px',
          fontWeight: 600,
          fontSize: '0.88rem',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.4rem',
        }}
      >
        <span>📂 Valuation History</span>
        {historyCount > 0 && (
          <span style={{
            backgroundColor: '#e6f4ea',
            color: '#137333',
            fontSize: '0.75rem',
            padding: '1px 6px',
            borderRadius: '10px'
          }}>
            {historyCount}
          </span>
        )}
      </button>

      {/* Footer Info */}
      <div style={{ marginTop: 'auto', fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
        <p><strong>Urbanscapes-LK</strong> MERN Stack</p>
        <p>Model Engine: Python XGBoost Regressor</p>
        <p>Database: MongoDB (Mongoose)</p>
      </div>
    </aside>
  );
}
