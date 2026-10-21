import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import L from 'leaflet';
import colomboBoundary from '../data/colomboBoundary.json';
import colomboSuburbsCoords from '../data/colomboSuburbsCoords.json';
import { isPointInDistrict, findNearestColomboSuburb } from '../utils/geoUtils';

export default function MapParcelPin({
  lat = 6.8770,
  lon = 79.9904,
  onCoordsChange,
  distFort = 14.89,
  isFloodRisk = false,
  suburb = '',
  suburbs = [],
  isMapSelected = false,
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);
  const [warning, setWarning] = useState(null);

  // Extract Colombo district boundary ring: array of [lat, lon]
  const colomboRing = useMemo(() => {
    return colomboBoundary.geometry.coordinates[0].map(([cLon, cLat]) => [cLat, cLon]);
  }, []);

  // Precise bounding box for Colombo District
  const colomboBounds = useMemo(() => {
    const southWest = L.latLng(6.712, 79.818);
    const northEast = L.latLng(6.985, 80.226);
    return L.latLngBounds(southWest, northEast);
  }, []);

  const fitDistrict = useCallback(() => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.fitBounds(colomboBounds, { padding: [12, 12] });
    }
  }, [colomboBounds]);

  const focusPin = useCallback(() => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([lat, lon], 14, { animate: true });
    }
  }, [lat, lon]);

  // Format popup HTML for marker
  const createPopupContent = useCallback((pLat, pLon, pArea, fromMap) => {
    const areaName = pArea ? (pArea.charAt(0).toUpperCase() + pArea.slice(1)) : 'Colombo';
    return `
      <div style="font-size:0.88rem; line-height:1.45; min-width: 175px;">
        <strong style="color:#0f172a; display:block; margin-bottom:2px;">📍 Pinned Parcel Location</strong>
        <div style="color:#1d4ed8; font-weight:700; margin-bottom:2px;">
          Area: ${areaName}
        </div>
        <div style="color:#64748b; font-size:0.8rem;">
          Lat: ${pLat.toFixed(4)}, Lon: ${pLon.toFixed(4)}
        </div>
        ${fromMap ? '<div style="color:#059669; font-size:0.75rem; font-weight:600; margin-top:3px;">✓ Selected on map</div>' : ''}
      </div>
    `;
  }, []);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Initialize map if not already initialized
    if (!mapInstanceRef.current) {
      // Initialize map strictly bounded to Colombo District
      const map = L.map(mapContainerRef.current, {
        center: [lat, lon],
        zoom: 11,
        minZoom: 11,
        maxZoom: 18,
        maxBounds: colomboBounds.pad(0.05), // Strictly prevent panning outside Colombo District
        maxBoundsViscosity: 1.0,           // Complete resistance to dragging outside bounds
        bounceAtZoomLimits: true,
      });

      // Standard OSM tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
      }).addTo(map);

      // Inverted polygon mask: dim all areas outside Colombo District
      const outerWorld = [
        [85, -180],
        [85, 180],
        [-85, 180],
        [-85, -180],
      ];
      L.polygon([outerWorld, colomboRing], {
        stroke: false,
        fillColor: '#0f172a',
        fillOpacity: 0.52,
        interactive: false,
      }).addTo(map);

      // Prominent crisp border along Colombo District boundary
      L.polygon(colomboRing, {
        color: '#1d4ed8',
        weight: 2.5,
        opacity: 0.95,
        dashArray: '5, 5',
        fillColor: '#2563eb',
        fillOpacity: 0.03,
        interactive: false,
      }).addTo(map);

      // Create pin marker with custom SVG icon
      const pinIcon = L.divIcon({
        className: 'custom-pin-marker',
        html: `
          <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            background-color: #FF4B4B;
            border-radius: 50% 50% 50% 0;
            transform: rotate(-45deg);
            box-shadow: 0 2px 6px rgba(0,0,0,0.35);
            border: 2px solid white;
          ">
            <div style="
              width: 10px;
              height: 10px;
              background-color: white;
              border-radius: 50%;
              transform: rotate(45deg);
            "></div>
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 32],
        popupAnchor: [0, -32],
      });

      const marker = L.marker([lat, lon], { icon: pinIcon }).addTo(map);
      marker.bindPopup(createPopupContent(lat, lon, suburb, isMapSelected)).openPopup();

      // Fit map view perfectly to Colombo District bounds
      map.fitBounds(colomboBounds, { padding: [12, 12] });

      // Click listener on map to move pin, strictly verified inside Colombo District
      map.on('click', (e) => {
        const clickLat = e.latlng.lat;
        const clickLon = e.latlng.lng;

        if (!isPointInDistrict(clickLat, clickLon, colomboRing)) {
          setWarning('⚠️ Selected point is outside Colombo District. Please pin a parcel within Colombo District boundaries.');
          L.popup()
            .setLatLng(e.latlng)
            .setContent(`
              <div style="font-size:0.85rem; font-weight:600; color:#dc2626; padding:2px;">
                ⚠️ Outside Colombo District<br/>
                <span style="font-weight:400; color:#4b5563; font-size:0.8rem;">
                  Please pin parcel within Colombo District boundary.
                </span>
              </div>
            `)
            .openOn(map);
          return;
        }

        setWarning(null);

        // Detect the nearest Colombo District suburb for the clicked coordinates
        const nearest = findNearestColomboSuburb(clickLat, clickLon, colomboSuburbsCoords, suburbs);
        const detectedSuburb = nearest?.suburb || suburb;

        marker.setLatLng([clickLat, clickLon]);
        marker.setPopupContent(createPopupContent(clickLat, clickLon, detectedSuburb, true)).openPopup();

        if (onCoordsChange) {
          onCoordsChange(clickLat, clickLon, detectedSuburb);
        }
      });

      mapInstanceRef.current = map;
      markerRef.current = marker;
    }

    return () => {
      // Cleanup on unmount
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [colomboBounds, colomboRing, createPopupContent]);

  // Update marker position & popup when coordinates or suburb change
  useEffect(() => {
    if (markerRef.current && mapInstanceRef.current) {
      markerRef.current.setLatLng([lat, lon]);
      markerRef.current.setPopupContent(createPopupContent(lat, lon, suburb, isMapSelected));

      // If marker is outside current viewport, pan to it smoothly
      if (!mapInstanceRef.current.getBounds().contains([lat, lon])) {
        mapInstanceRef.current.panTo([lat, lon], { animate: true });
      }
    }
  }, [lat, lon, suburb, isMapSelected, createPopupContent]);

  return (
    <div className="column-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <h2 className="section-title">📍 1. Interactive GIS Parcel Pinning (Leaflet/OSM)</h2>
            <span style={{
              backgroundColor: '#e0f2fe',
              color: '#0369a1',
              fontSize: '0.75rem',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: '12px',
              border: '1px solid #bae6fd',
              whiteSpace: 'nowrap',
            }}>
              Colombo District Only
            </span>
          </div>
          <p className="section-caption">
            Map strictly bounded to Colombo District. Click anywhere inside the boundary to pin a parcel.
          </p>
        </div>

        {/* View Controls */}
        <div style={{ display: 'flex', gap: '0.35rem' }}>
          <button
            type="button"
            onClick={fitDistrict}
            title="Reset map view to show entire Colombo District"
            style={{
              fontSize: '0.78rem',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #cbd5e1',
              backgroundColor: '#ffffff',
              color: '#334155',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            🎯 District View
          </button>
          <button
            type="button"
            onClick={focusPin}
            title="Zoom into the pinned parcel"
            style={{
              fontSize: '0.78rem',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #cbd5e1',
              backgroundColor: '#ffffff',
              color: '#334155',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            📍 Focus Pin
          </button>
        </div>
      </div>

      {/* Map Box */}
      <div ref={mapContainerRef} className="map-container-box" />

      {/* Outside District Alert */}
      {warning && (
        <div className="st-alert st-alert-error">
          {warning}
        </div>
      )}

      {/* Info Alert Box */}
      <div className="st-alert st-alert-info">
        <strong>Selected Area:</strong>{' '}
        <span style={{ textTransform: 'capitalize', fontWeight: 700, color: '#1e40af' }}>
          {suburb}
        </span>
        {isMapSelected && (
          <span
            style={{
              fontSize: '0.74rem',
              backgroundColor: '#dbeafe',
              color: '#1d4ed8',
              fontWeight: 600,
              padding: '1px 6px',
              borderRadius: '4px',
              marginLeft: '6px',
              marginRight: '6px',
            }}
          >
            Selected on map
          </span>
        )}{' '}
        | <strong>Coordinates:</strong> Lat <code>{lat.toFixed(4)}</code>, Lon{' '}
        <code>{lon.toFixed(4)}</code> | <strong>Distance to Fort:</strong> <code>{distFort} km</code>
      </div>

      {/* Flood Hazard Alert Box */}
      {isFloodRisk ? (
        <div className="st-alert st-alert-error">
          ⚠️ <strong>NBRO Environmental Hazard Flag:</strong> High Flood Vulnerability Zone Detected (-10% Risk Adjustment Applied)
        </div>
      ) : (
        <div className="st-alert st-alert-success">
          ✅ <strong>NBRO Environmental Hazard Flag:</strong> Low Risk Zone (Safe for Development)
        </div>
      )}
    </div>
  );
}
