# Urban-Growth-Optimizer_IT-337
We are building an automated spatial growth prediction, unauthorized building detection, automated lands valuation, and infrastructure optimization system for the Urban Development Authority of Sri Lanka.

---

## Component 4: Multi-Criteria Infrastructure Spatial Optimization DSS

A GIS-based Spatial Decision Support System (SDSS) integrating Analytic Hierarchy Process (AHP), Weighted Linear Combination (WLC), and dual Metaheuristic Optimization (GA vs. PSO) to identify optimal public facility sites (schools, hospitals) in Matale District.

### Key Features
- **AHP Engine**: Saaty eigenvector weighting across 8 environmental & demographic criteria with Consistency Ratio validation ($CR < 0.032$).
- **Dual Metaheuristic Optimizers**: Genetic Algorithm (GA) vs. Particle Swarm Optimization (PSO) for continuous spatial site discovery.
- **Real GIS Data Ingestion**:
  - NASA SRTM 30m DEM (elevation & metric slope)
  - WorldPop 100m raster fused with 63,860 OSM building footprints
  - OSM Sri Lanka road network (11,116 segments)
  - 749 waterways with statutory flood buffers (60m rivers, 30m canals)
  - 125 religious/cultural heritage precinct buffers (30m)
- **NBRO & CEA Hard Constraint Masking**: Automatic disqualification with legal penalty (`-999.0`).
- **Reverse HBU Siting & Building Spec Predictor**: Max recommended floors, Solar PV potential (kWp), and estimated construction budget.
- **Interactive 2D & 3D WebGL Dashboard**: Leaflet GIS map with Three.js 3D terrain viewer on port 8050.
- **Empirical Validation**: Ground-truth validation against existing Matale facilities (Hospital AUC: 88.24%, School AUC: 79.20%, Mann-Whitney $p < 0.001$).

### Setup & Installation

```powershell
cd spatial_optimization
pip install -r requirements.txt
```

### Running the Interactive Dashboard

```powershell
python interactive_dashboard.py
```
Then navigate to `http://localhost:8050` in your web browser.

### Running Unit Tests

```powershell
pytest tests/ -v
```
