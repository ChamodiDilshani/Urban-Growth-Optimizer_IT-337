/**
 * GIS utilities for Colombo District
 */

// Ray-casting point-in-polygon algorithm to check if coordinate is inside Colombo District boundary
export function isPointInDistrict(lat, lon, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1]; // lat, lon
    const xj = ring[j][0], yj = ring[j][1];
    const intersect = ((yi > lon) !== (yj > lon)) &&
      (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

// Find nearest known Colombo suburb using Haversine geodesic distance formula
export function findNearestColomboSuburb(lat, lon, suburbsCoordsMap, availableSuburbs = []) {
  let nearestKey = null;
  let minDistance = Infinity;
  const R = 6371.0; // Earth radius in km

  for (const [key, item] of Object.entries(suburbsCoordsMap)) {
    const dLat = (item.lat - lat) * (Math.PI / 180);
    const dLon = (item.lon - lon) * (Math.PI / 180);
    const lat1 = lat * (Math.PI / 180);
    const lat2 = item.lat * (Math.PI / 180);

    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const dist = R * c;

    if (dist < minDistance) {
      minDistance = dist;
      nearestKey = key;
    }
  }

  if (!nearestKey) return null;

  // Match with exact casing in availableSuburbs if provided
  let matchedSuburb = suburbsCoordsMap[nearestKey]?.name || nearestKey;
  if (availableSuburbs.length > 0) {
    const exact = availableSuburbs.find(
      (s) => s.toLowerCase().trim() === nearestKey.toLowerCase().trim()
    );
    if (exact) {
      matchedSuburb = exact;
    }
  }

  return {
    suburb: matchedSuburb,
    key: nearestKey,
    distanceKm: Math.round(minDistance * 100) / 100,
  };
}
