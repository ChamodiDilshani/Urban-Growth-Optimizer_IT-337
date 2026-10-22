/**
 * NBRO (National Building Research Organisation) Flood Vulnerability & Hazard Engine
 * Urban Growth Optimizer IT-337
 * 
 * Provides spatial (lat/lon) and municipal boundary hazard detection for Colombo District,
 * including low-lying retention wetlands (e.g. SLIIT Malabe / Kothalawala wetland corridor,
 * Kelani River basin, Diyawanna wetland, and Bellanwila-Attidiya sanctuary).
 */

export const NBRO_FLOOD_ZONES = [
  {
    id: 'sliit_malabe',
    name: 'Malabe (SLIIT - Kothalawala Wetland Corridor)',
    category: 'Low-Lying Wetland & Inundation Basin',
    center: [6.9145, 79.9728],
    radiusKm: 1.35,
    polygon: [
      [6.9240, 79.9680],
      [6.9260, 79.9770],
      [6.9190, 79.9830],
      [6.9100, 79.9800],
      [6.9060, 79.9710],
      [6.9080, 79.9640],
      [6.9170, 79.9650],
    ],
  },
  {
    id: 'kaduwela_kelani',
    name: 'Kaduwela - Nawagamuwa (Kelani River Basin)',
    category: 'Riverine Inundation Plain',
    center: [6.9330, 79.9840],
    radiusKm: 1.80,
    polygon: [
      [6.9450, 79.9700],
      [6.9480, 79.9950],
      [6.9280, 80.0100],
      [6.9200, 79.9850],
      [6.9250, 79.9680],
    ],
  },
  {
    id: 'kolonnawa_wellampitiya',
    name: 'Kolonnawa - Wellampitiya - Sedawatta Basin',
    category: 'Urban Drainage & Canal Basin',
    center: [6.9400, 79.8880],
    radiusKm: 1.75,
    polygon: [
      [6.9560, 79.8700],
      [6.9580, 79.8980],
      [6.9380, 79.9120],
      [6.9260, 79.8900],
      [6.9340, 79.8720],
    ],
  },
  {
    id: 'kotikawatta_mulleriyawa',
    name: 'Kotikawatta - Mulleriyawa - Angoda Basin',
    category: 'Kelani Flood Retention Basin',
    center: [6.9320, 79.9180],
    radiusKm: 1.60,
    polygon: [
      [6.9450, 79.9050],
      [6.9480, 79.9320],
      [6.9250, 79.9360],
      [6.9180, 79.9150],
    ],
  },
  {
    id: 'ambatale_kelanimulla',
    name: 'Ambatale - Kelanimulla Corridor',
    category: 'River Bank Spill Zone',
    center: [6.9460, 79.9400],
    radiusKm: 1.40,
    polygon: [
      [6.9550, 79.9280],
      [6.9580, 79.9550],
      [6.9380, 79.9560],
      [6.9350, 79.9300],
    ],
  },
  {
    id: 'bomiriya_ranala',
    name: 'Bomiriya - Ranala - Hanwella Floodplain',
    category: 'Upper Kelani River Spillway',
    center: [6.9300, 80.0250],
    radiusKm: 2.20,
    polygon: [
      [6.9450, 79.9980],
      [6.9480, 80.0550],
      [6.9150, 80.0600],
      [6.9120, 80.0100],
    ],
  },
  {
    id: 'diyawanna_parliament',
    name: 'Diyawanna - Parliament Marsh Retention Basin',
    category: 'Protected Urban Wetland Retention Zone',
    center: [6.8960, 79.9130],
    radiusKm: 1.30,
    polygon: [
      [6.9120, 79.9020],
      [6.9140, 79.9240],
      [6.8850, 79.9280],
      [6.8820, 79.9060],
    ],
  },
  {
    id: 'bellanwila_attidiya',
    name: 'Bellanwila - Attidiya Sanctuary Basin',
    category: 'Weras Ganga Wetland & Inundation Basin',
    center: [6.8450, 79.8900],
    radiusKm: 1.25,
    polygon: [
      [6.8580, 79.8800],
      [6.8600, 79.9020],
      [6.8320, 79.9040],
      [6.8300, 79.8820],
    ],
  },
  {
    id: 'bolgoda_north',
    name: 'Bolgoda Lake Northern Basin (Dampe / Kesbewa)',
    category: 'Bolgoda Basin Lowland Fringe',
    center: [6.8020, 79.9220],
    radiusKm: 1.20,
    polygon: [
      [6.8150, 79.9100],
      [6.8160, 79.9350],
      [6.7900, 79.9380],
      [6.7880, 79.9120],
    ],
  },
];

// Municipal suburbs that are recognized as flood-prone by default
export const INHERENT_FLOOD_SUBURBS = [
  'kaduwela',
  'ranala',
  'bomiriya',
  'kolonnawa',
  'wellampitiya',
  'kotikawatta',
  'angoda',
  'kelanimulla',
  'ambathale',
  'ambatalemulla',
  'hanwella',
  'nawagamuwa',
  'sedawatta',
  'mulleriyawa',
  'meetotamulla',
];

/**
 * Standard ray-casting algorithm to test if a point is within a polygon.
 */
export function isPointInPolygon(lat, lon, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];

    const intersect = yi > lon !== yj > lon && lat < ((xj - xi) * (lon - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/**
 * Geodesic distance calculation in kilometers.
 */
export function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Evaluates flood hazard status given GPS coordinates (lat, lon) and optional suburb name.
 * Checks spatial polygon boundaries, radius proximity buffers, and municipal listings.
 */
export function evaluateFloodRisk(lat, lon, suburbName = '') {
  if (typeof lat !== 'number' || typeof lon !== 'number') {
    return {
      isFloodRisk: false,
      zoneName: null,
      category: null,
      distanceToZoneKm: null,
    };
  }

  // 1. Check spatial polygon intersection first (most precise)
  for (const zone of NBRO_FLOOD_ZONES) {
    if (zone.polygon && isPointInPolygon(lat, lon, zone.polygon)) {
      return {
        isFloodRisk: true,
        zoneName: zone.name,
        category: zone.category,
        distanceToZoneKm: 0.0,
      };
    }
  }

  // 2. Check circular proximity buffer around zone centers
  for (const zone of NBRO_FLOOD_ZONES) {
    const dist = haversineDistance(lat, lon, zone.center[0], zone.center[1]);
    if (dist <= zone.radiusKm) {
      return {
        isFloodRisk: true,
        zoneName: zone.name,
        category: zone.category,
        distanceToZoneKm: Number(dist.toFixed(2)),
      };
    }
  }

  // 3. Fallback: Municipal suburb level match
  const cleanSuburb = String(suburbName).toLowerCase().trim();
  if (INHERENT_FLOOD_SUBURBS.includes(cleanSuburb)) {
    return {
      isFloodRisk: true,
      zoneName: `${suburbName.charAt(0).toUpperCase() + suburbName.slice(1)} (Kelani Basin Suburb)`,
      category: 'Municipal Flood Hazard Zone',
      distanceToZoneKm: 0.0,
    };
  }

  // Safe / Low Risk Zone
  return {
    isFloodRisk: false,
    zoneName: null,
    category: 'Low Risk Zone',
    distanceToZoneKm: null,
  };
}
