import React, { useState, useRef, useEffect, useMemo } from 'react';

export default function SearchableSelect({
  options = [],
  value = '',
  onChange,
  placeholder = 'Select a suburb...',
  id = 'searchable-suburb-select',
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [highlightedIndex, setHighlightedIndex] = useState(-1);

  const containerRef = useRef(null);
  const searchInputRef = useRef(null);
  const listRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Filter options based on search query
  const filteredOptions = useMemo(() => {
    if (!searchTerm.trim()) return options;
    const term = searchTerm.toLowerCase().trim();
    return options.filter((opt) => opt.toLowerCase().includes(term));
  }, [options, searchTerm]);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (isOpen) {
      setHighlightedIndex(-1);
      setTimeout(() => {
        searchInputRef.current?.focus();
      }, 50);
    } else {
      setSearchTerm('');
    }
  }, [isOpen]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (highlightedIndex >= 0 && listRef.current) {
      const items = listRef.current.querySelectorAll('.searchable-option-item');
      if (items[highlightedIndex]) {
        items[highlightedIndex].scrollIntoView({ block: 'nearest' });
      }
    }
  }, [highlightedIndex]);

  const handleSelect = (suburb) => {
    if (onChange) {
      onChange(suburb);
    }
    setIsOpen(false);
  };

  const handleKeyDown = (e) => {
    if (!isOpen) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
        e.preventDefault();
        setIsOpen(true);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex((prev) =>
        prev < filteredOptions.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex((prev) =>
        prev > 0 ? prev - 1 : filteredOptions.length - 1
      );
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightedIndex >= 0 && filteredOptions[highlightedIndex]) {
        handleSelect(filteredOptions[highlightedIndex]);
      } else if (filteredOptions.length === 1) {
        handleSelect(filteredOptions[0]);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setIsOpen(false);
    }
  };

  // Helper to highlight matching text
  const highlightMatch = (text, query) => {
    if (!query.trim()) return text;
    const q = query.toLowerCase();
    const idx = text.toLowerCase().indexOf(q);
    if (idx === -1) return text;
    return (
      <>
        {text.slice(0, idx)}
        <mark style={{ backgroundColor: '#fef08a', color: '#854d0e', padding: '0 1px', borderRadius: '2px', fontWeight: 700 }}>
          {text.slice(idx, idx + q.length)}
        </mark>
        {text.slice(idx + q.length)}
      </>
    );
  };

  const displayValue = value
    ? value.charAt(0).toUpperCase() + value.slice(1)
    : placeholder;

  return (
    <div
      ref={containerRef}
      className="searchable-select-container"
      style={{ position: 'relative', width: '100%' }}
      onKeyDown={handleKeyDown}
    >
      {/* Trigger Button */}
      <button
        id={id}
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="form-select searchable-trigger"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          textAlign: 'left',
          cursor: 'pointer',
          padding: '0.55rem 0.75rem',
          backgroundColor: '#ffffff',
          borderRadius: '6px',
          border: isOpen ? '1px solid var(--primary-color, #ff4b4b)' : '1px solid #d3d7df',
          boxShadow: isOpen ? '0 0 0 2px rgba(255, 75, 75, 0.2)' : 'none',
          userSelect: 'none',
        }}
      >
        <span style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          color: value ? 'var(--text-primary)' : 'var(--text-muted)',
          fontWeight: value ? 600 : 400,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          <span style={{ fontSize: '0.9rem' }}>📍</span>
          <span>{displayValue}</span>
        </span>
        <span style={{
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
          marginLeft: '0.5rem',
          transition: 'transform 0.2s',
          transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
        }}>
          ▼
        </span>
      </button>

      {/* Floating Dropdown Panel */}
      {isOpen && (
        <div
          className="searchable-dropdown-menu"
          style={{
            position: 'absolute',
            top: 'calc(100% + 4px)',
            left: 0,
            right: 0,
            backgroundColor: '#ffffff',
            border: '1px solid #cbd5e1',
            borderRadius: '8px',
            boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            animation: 'fadeIn 0.15s ease-out',
          }}
        >
          {/* Search Box Header */}
          <div style={{
            padding: '8px',
            backgroundColor: '#f8fafc',
            borderBottom: '1px solid #e2e8f0',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <span style={{
                position: 'absolute',
                left: '9px',
                color: '#94a3b8',
                fontSize: '0.85rem',
                pointerEvents: 'none',
              }}>
                🔍
              </span>
              <input
                ref={searchInputRef}
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Type to search suburb (e.g. Pinnawala, Maharagama)..."
                style={{
                  width: '100%',
                  padding: '6px 28px 6px 30px',
                  fontSize: '0.86rem',
                  border: '1px solid #cbd5e1',
                  borderRadius: '5px',
                  outline: 'none',
                  backgroundColor: '#ffffff',
                }}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter') {
                    // Let parent container handle navigation
                  }
                }}
              />
              {searchTerm && (
                <button
                  type="button"
                  onClick={() => setSearchTerm('')}
                  style={{
                    position: 'absolute',
                    right: '6px',
                    background: 'none',
                    border: 'none',
                    color: '#94a3b8',
                    cursor: 'pointer',
                    fontSize: '0.85rem',
                    padding: '2px 6px',
                  }}
                  title="Clear search"
                >
                  ✕
                </button>
              )}
            </div>

            {/* Suburb count indicator */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.72rem',
              color: '#64748b',
              padding: '2px 4px 0 4px',
            }}>
              <span>
                {filteredOptions.length === options.length
                  ? `Showing all ${options.length} Colombo suburbs`
                  : `${filteredOptions.length} matching area${filteredOptions.length === 1 ? '' : 's'} found`}
              </span>
              {searchTerm && (
                <span style={{ fontStyle: 'italic' }}>Press Enter to select</span>
              )}
            </div>
          </div>

          {/* Scrollable Suburbs List */}
          <div
            ref={listRef}
            style={{
              maxHeight: '230px',
              overflowY: 'auto',
              padding: '4px 0',
            }}
          >
            {filteredOptions.length > 0 ? (
              filteredOptions.map((suburb, index) => {
                const isSelected =
                  suburb.toLowerCase().trim() === (value || '').toLowerCase().trim();
                const isHighlighted = index === highlightedIndex;

                return (
                  <div
                    key={suburb}
                    className="searchable-option-item"
                    onClick={() => handleSelect(suburb)}
                    onMouseEnter={() => setHighlightedIndex(index)}
                    style={{
                      padding: '7px 12px',
                      fontSize: '0.88rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      backgroundColor: isHighlighted
                        ? '#f1f5f9'
                        : isSelected
                        ? '#eff6ff'
                        : 'transparent',
                      color: isSelected ? '#1d4ed8' : '#0f172a',
                      fontWeight: isSelected ? 700 : 400,
                      transition: 'background-color 0.1s',
                    }}
                  >
                    <span>
                      {highlightMatch(
                        suburb.charAt(0).toUpperCase() + suburb.slice(1),
                        searchTerm
                      )}
                    </span>
                    {isSelected && (
                      <span style={{ color: '#2563eb', fontSize: '0.85rem' }}>✓</span>
                    )}
                  </div>
                );
              })
            ) : (
              <div style={{
                padding: '16px',
                textAlign: 'center',
                color: '#64748b',
                fontSize: '0.85rem',
              }}>
                <div style={{ fontSize: '1.2rem', marginBottom: '4px' }}>🔍</div>
                <div>No suburbs found matching "<strong>{searchTerm}</strong>"</div>
                <button
                  type="button"
                  onClick={() => setSearchTerm('')}
                  style={{
                    marginTop: '8px',
                    padding: '3px 8px',
                    fontSize: '0.78rem',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    backgroundColor: '#ffffff',
                    cursor: 'pointer',
                    color: '#2563eb',
                  }}
                >
                  Clear search filter
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
