import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MapParcelPin from './components/MapParcelPin';
import MetricsDisplay from './components/MetricsDisplay';
import ShapExplainer from './components/ShapExplainer';
import HistoryModal from './components/HistoryModal';
import { api } from './services/api';
import colomboSuburbsCoords from './data/colomboSuburbsCoords.json';

export default function App() {
  // Metadata from backend
  const [metadata, setMetadata] = useState({
    suburbs: [],
    land_types: [],
    road_widths: ['10 ft', '12 ft', '15 ft', '20 ft', '30+ ft'],
    flood_prone: [],
  });
  const [dbConnected, setDbConnected] = useState(false);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(true);
  const [backendError, setBackendError] = useState(null);

  // Form parameters (matching Streamlit defaults with validated Colombo coordinates)
  const [params, setParams] = useState({
    suburb: 'athurugiriya',
    land_size: 12.0,
    land_type: 'Residential',
    road_width: '15 ft',
    target_year: 2026,
    lat: 6.8770,
    lon: 79.9904,
  });

  // Output from model inference
  const [valuation, setValuation] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

  // History modal
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [historySource, setHistorySource] = useState('memory');

  const debounceTimerRef = useRef(null);

  // Suburb change handler (auto-centers map to suburb coordinates within Colombo district)
  const handleSuburbChange = useCallback((newSuburb) => {
    const key = newSuburb.toLowerCase().trim();
    const coord = colomboSuburbsCoords[key];
    setParams((prev) => ({
      ...prev,
      suburb: newSuburb,
      ...(coord ? { lat: coord.lat, lon: coord.lon } : {}),
    }));
  }, []);

  // 1. Initial Metadata Load
  useEffect(() => {
    async function loadMeta() {
      try {
        setIsLoadingMetadata(true);
        const res = await api.getMetadata();
        if (res.success && res.data) {
          setMetadata(res.data);
          setDbConnected(Boolean(res.dbConnected));

          // Set default suburb if not already set or invalid
          if (res.data.suburbs?.length > 0) {
            setParams((p) => {
              const activeSub = res.data.suburbs.includes(p.suburb) ? p.suburb : res.data.suburbs[0];
              const key = activeSub.toLowerCase().trim();
              const coord = colomboSuburbsCoords[key];
              return {
                ...p,
                suburb: activeSub,
                land_type: res.data.land_types.includes(p.land_type) ? p.land_type : res.data.land_types[0],
                ...(coord ? { lat: coord.lat, lon: coord.lon } : {}),
              };
            });
          }
        }
      } catch (err) {
        console.error('Failed to load initial metadata:', err);
        setBackendError('Could not connect to the Express Backend. Please ensure the backend and Python service are running.');
      } finally {
        setIsLoadingMetadata(false);
      }
    }
    loadMeta();
    fetchHistory();
  }, []);

  // 2. Fetch History
  const fetchHistory = async () => {
    try {
      const res = await api.getHistory();
      if (res.success) {
        setHistoryList(res.data || []);
        setHistorySource(res.source || 'memory');
      }
    } catch (err) {
      console.warn('Could not load history:', err.message);
    }
  };

  // 3. Trigger Model Inference (Debounced)
  const executePrediction = useCallback(async (currentParams) => {
    if (!currentParams.suburb) return;
    setIsPredicting(true);
    setBackendError(null);

    try {
      const res = await api.predict(currentParams);
      if (res.success) {
        setValuation(res.data);
        setDbConnected(Boolean(res.dbConnected));
        fetchHistory(); // Refresh history count
      }
    } catch (err) {
      console.error('Inference error:', err);
      setBackendError(err.message || 'Model prediction failed.');
    } finally {
      setIsPredicting(false);
    }
  }, []);

  // Auto-predict on parameter change with 400ms debounce
  useEffect(() => {
    if (isLoadingMetadata) return;

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      executePrediction(params);
    }, 450);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [params, isLoadingMetadata, executePrediction]);

  // Coordinate change callback from map pin
  const handleCoordsChange = (newLat, newLon) => {
    setParams((prev) => ({
      ...prev,
      lat: newLat,
      lon: newLon,
    }));
  };

  // 4. Download PDF (In-Memory Base64 download to completely bypass IDM interception)
  const handleDownloadPdf = async () => {
    if (!valuation) return;

    try {
      setIsDownloadingPdf(true);
      const filename = valuation.pdf_filename || `Assessment_Tax_${params.suburb}_${params.target_year}.pdf`;

      // If already computed with prediction, trigger immediate in-memory download
      if (valuation.pdf_base64) {
        const link = document.createElement('a');
        link.href = `data:application/pdf;base64,${valuation.pdf_base64}`;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        return;
      }

      // Fallback: request base64 JSON from backend
      const isFlood = Boolean(valuation.is_flood_risk);

      const parcelInfo = {
        'Suburb / Area': params.suburb.charAt(0).toUpperCase() + params.suburb.slice(1),
        'Coordinates': `${params.lat.toFixed(4)}, ${params.lon.toFixed(4)}`,
        'Land Extent': `${params.land_size} Perches`,
        'Zoning Category': params.land_type,
        'Access Road Width': params.road_width,
        'Distance to Colombo Fort': `${valuation.dist_fort} km`,
        'NBRO Hazard Status': isFlood ? 'High Flood Risk' : 'Low Risk Zone',
      };

      const valInfo = {
        'Predicted Price per Perch': `LKR ${Number(valuation.final_price_per_perch).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
        'Total Parcel Valuation': `LKR ${Number(valuation.total_valuation).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
        'Annual Assessed Rental Value': `LKR ${Number(valuation.annual_assessed_value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
        [`Statutory Tax (${params.target_year})`]: `LKR ${Number(valuation.annual_tax).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
        'Multi-Year': valuation.multi_year_tax || {},
      };

      await api.downloadPdf(parcelInfo, valInfo, filename);
    } catch (err) {
      alert(`PDF download failed: ${err.message}`);
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  // Restore history record
  const handleRestoreHistory = (item) => {
    setParams({
      suburb: item.suburb,
      land_size: item.landSize,
      land_type: item.landType,
      road_width: item.roadWidth,
      target_year: item.targetYear,
      lat: item.coordinates?.lat || 6.9022,
      lon: item.coordinates?.lon || 79.9172,
    });
    setIsHistoryOpen(false);
  };

  const handleDeleteHistory = async (id) => {
    try {
      await api.deleteHistoryItem(id);
      fetchHistory();
    } catch (err) {
      console.error(err);
    }
  };

  const handleClearHistory = async () => {
    if (window.confirm('Are you sure you want to clear all valuation history?')) {
      try {
        await api.clearHistory();
        fetchHistory();
      } catch (err) {
        console.error(err);
      }
    }
  };

  return (
    <div className="streamlit-container">
      {/* Streamlit Sidebar */}
      <Sidebar
        suburbs={metadata.suburbs}
        landTypes={metadata.land_types}
        roadWidths={metadata.road_widths}
        params={params}
        setParams={setParams}
        onOpenHistory={() => setIsHistoryOpen(true)}
        historyCount={historyList.length}
        onSuburbChange={handleSuburbChange}
      />

      {/* Main Canvas Area */}
      <main className="streamlit-main">
        <Header dbConnected={dbConnected} isPredicting={isPredicting} />

        {backendError && (
          <div className="st-alert st-alert-error" style={{ marginBottom: '1.5rem' }}>
            ⚠️ <strong>Backend Alert:</strong> {backendError}
          </div>
        )}

        {/* 2-Column Responsive Layout matching Streamlit col1, col2 = st.columns([1.1, 1]) */}
        <div className="content-grid">
          {/* Column 1: Interactive GIS Parcel Pinning */}
          <MapParcelPin
            lat={params.lat}
            lon={params.lon}
            onCoordsChange={handleCoordsChange}
            distFort={valuation?.dist_fort ?? 14.89}
            isFloodRisk={valuation?.is_flood_risk ?? false}
            suburb={params.suburb}
          />

          {/* Column 2: Valuation, Tax Metrics, PDF Download & SHAP */}
          <div className="column-panel">
            <MetricsDisplay
              valuation={valuation}
              targetYear={params.target_year}
              onDownloadPdf={handleDownloadPdf}
              isDownloadingPdf={isDownloadingPdf}
            />

            <ShapExplainer
              shapImage={valuation?.shap_image}
              isLoading={isPredicting}
            />
          </div>
        </div>
      </main>

      {/* Valuation History Modal (MongoDB) */}
      <HistoryModal
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={historyList}
        onRestore={handleRestoreHistory}
        onDelete={handleDeleteHistory}
        onClearAll={handleClearHistory}
        historySource={historySource}
      />
    </div>
  );
}
