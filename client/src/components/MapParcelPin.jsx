import React, { useEffect, useRef } from 'react';
import L from 'leaflet';

export default function MapParcelPin({
  lat = 6.9022,
  lon = 79.9172,
  onCoordsChange,
  distFort = 14.89,
  isFloodRisk = false,
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Initialize map if not already initialized
    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [lat, lon],
        zoom: 11,
      });

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
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
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
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
      marker.bindPopup(`<b>Pinned Parcel Location</b><br/>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`).openPopup();

      // Click listener on map to move pin
      map.on('click', (e) => {
        const newLat = e.latlng.lat;
        const newLon = e.latlng.lng;
        marker.setLatLng([newLat, newLon]);
        marker.setPopupContent(`<b>Pinned Parcel Location</b><br/>Lat: ${newLat.toFixed(4)}, Lon: ${newLon.toFixed(4)}`).openPopup();
        if (onCoordsChange) {
          onCoordsChange(newLat, newLon);
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
  }, []);

  // Update marker position if coordinates change externally
  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.setLatLng([lat, lon]);
      markerRef.current.setPopupContent(`<b>Pinned Parcel Location</b><br/>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`);
    }
  }, [lat, lon]);

  return (
    <div className="column-panel">
      <div>
        <h2 className="section-title">📍 1. Interactive GIS Parcel Pinning (Leaflet/OSM)</h2>
        <p className="section-caption">
          Click anywhere on the map to pin the land parcel and compute geodesic distances via Haversine formula.
        </p>
      </div>

      {/* Map Box */}
      <div ref={mapContainerRef} className="map-container-box" />

      {/* Info Alert Box */}
      <div className="st-alert st-alert-info">
        <strong>Pinned Coordinates:</strong> Lat <code>{lat.toFixed(4)}</code>, Lon <code>{lon.toFixed(4)}</code> | <strong>Distance to Fort:</strong> <code>{distFort} km</code>
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
