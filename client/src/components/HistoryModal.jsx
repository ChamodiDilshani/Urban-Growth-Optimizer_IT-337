import React from 'react';

export default function HistoryModal({
  isOpen,
  onClose,
  history = [],
  onRestore,
  onDelete,
  onClearAll,
  historySource,
}) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h3 className="modal-title">📂 Saved Valuation History</h3>
            <span
              style={{
                fontSize: '0.75rem',
                padding: '2px 8px',
                borderRadius: '12px',
                backgroundColor: historySource === 'mongodb' ? '#e6f4ea' : '#fef7e0',
                color: historySource === 'mongodb' ? '#137333' : '#b06000',
                fontWeight: 600,
              }}
            >
              {historySource === 'mongodb' ? '🍃 Stored in MongoDB' : '💾 In-Memory Cache'}
            </span>
          </div>
          <button className="modal-close-btn" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="modal-body">
          {history.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
              No valuations recorded yet. Perform a valuation calculation to see it saved here!
            </div>
          ) : (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Total records: <strong>{history.length}</strong>
                </span>
                <button
                  type="button"
                  onClick={onClearAll}
                  style={{
                    fontSize: '0.8rem',
                    color: '#c5221f',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    fontWeight: 600,
                  }}
                >
                  Clear All History
                </button>
              </div>

              {history.map((item) => (
                <div key={item._id} className="history-item">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <strong style={{ fontSize: '1rem', color: 'var(--text-primary)' }}>
                        {item.suburb}
                      </strong>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        ({item.landSize} Perches • {item.landType})
                      </span>
                    </div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                      Price/Perch: <strong>LKR {Number(item.finalPricePerPerch).toLocaleString('en-US', { maximumFractionDigits: 0 })}</strong> | Total: <strong>LKR {Number(item.totalValuation).toLocaleString('en-US', { maximumFractionDigits: 0 })}</strong>
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Date: {new Date(item.createdAt).toLocaleString()} | Horizon: {item.targetYear}
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <button
                      type="button"
                      onClick={() => onRestore(item)}
                      style={{
                        padding: '5px 10px',
                        fontSize: '0.82rem',
                        fontWeight: 600,
                        backgroundColor: '#f1f3f4',
                        border: '1px solid #dadce0',
                        borderRadius: '4px',
                        cursor: 'pointer',
                      }}
                    >
                      Load
                    </button>
                    <button
                      type="button"
                      onClick={() => onDelete(item._id)}
                      style={{
                        padding: '5px 8px',
                        fontSize: '0.82rem',
                        color: '#d93025',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                      }}
                      title="Delete record"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
