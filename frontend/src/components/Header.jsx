import React from 'react';

export default function Header({ dbConnected, isPredicting }) {
  return (
    <header className="app-header">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h1 className="app-title">🏙️ Urbanscapes-LK: Automated Property Valuation & Assessment Tax Engine</h1>
          <p className="app-subtitle">
            <strong>Component 3:</strong> AI-Driven Land Valuation, SHAP Explainability & Municipal Assessment Tax (වරිපනම් බදු) Forecasting
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{
            fontSize: '0.78rem',
            padding: '3px 8px',
            borderRadius: '12px',
            fontWeight: 600,
            backgroundColor: '#e6f4ea',
            color: '#137333',
            border: '1px solid #ceead6'
          }}>
            ⚡ Python ML Active
          </span>
          <span style={{
            fontSize: '0.78rem',
            padding: '3px 8px',
            borderRadius: '12px',
            fontWeight: 600,
            backgroundColor: dbConnected ? '#e6f4ea' : '#fef7e0',
            color: dbConnected ? '#137333' : '#b06000',
            border: `1px solid ${dbConnected ? '#ceead6' : '#fce8b2'}`
          }}>
            {dbConnected ? '🍃 MongoDB Connected' : '💾 In-Memory Cache (MongoDB Available)'}
          </span>
          {isPredicting && (
            <span style={{
              fontSize: '0.78rem',
              padding: '3px 8px',
              borderRadius: '12px',
              fontWeight: 600,
              backgroundColor: '#e8f0fe',
              color: '#1a73e8',
            }}>
              ⏳ Computing...
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
