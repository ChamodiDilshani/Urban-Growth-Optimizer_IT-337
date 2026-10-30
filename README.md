# Urban-Growth-Optimizer_IT-337
## Intelligent Spatial Decision Support Platform for Urban Planning & Infrastructure Management

**Faculty of Computing | Department of Information Technology**  
**Sri Lanka Institute of Information Technology (SLIIT)**  
**IT4010 Research Project**

---

### Research Metadata
- **Project ID:** TAF_J26 – IT – 337
- **Component 4:** Multi-Criteria Spatial Optimization for Infrastructure Site Selection
- **Student Registration Number:** IT23170902
- **Research Group:** CoEAI – Centre of Excellence for AI
- **Supervisors:** Ms. Lokesha Weerasinghe (Supervisor), Ms. Ishara Weerathunga (Co-Supervisor), Mrs. Chamila Ekanayake (External Supervisor - UDA)
- **Target Area:** Matale District, Central Province, Sri Lanka

---

## 📌 Project Overview
Component 4 of the **UrbanScapes-LK** platform is an autonomous Spatial Decision Support System (SDSS) designed to eliminate manual guesswork and unscientific political allocations in public infrastructure placement (schools, hospitals, transit hubs).

The engine integrates multi-criteria decision analysis (**Saaty's Analytic Hierarchy Process - AHP**), spatial raster modeling (**Weighted Linear Combination - WLC**), statutory environmental constraint masking (NBRO & Central Environmental Authority guidelines), and continuous dual metaheuristic optimization (**Particle Swarm Optimization - PSO** vs. **Genetic Algorithm - GA**).

---

## 🚀 Key Features

1. **AHP Multi-Criteria Engine:**
   - Pairwise Comparison Matrix ($8 \times 8$) covering elevation, slope, road proximity, demographic demand, flood risk, land use, transport hubs, and amenity buffers.
   - Strict Saaty Consistency Ratio enforcement ($CR < 0.032$, well below the $0.10$ threshold).

2. **Dual Metaheuristic Spatial Optimizers:**
   - **Particle Swarm Optimization (PSO):** Swarm intelligence with velocity update and inertia weight decay for rapid continuous $(x, y)$ spatial siting.
   - **Genetic Algorithm (GA):** Tournament selection, arithmetic crossover, and Gaussian mutation.
   - **Benchmark:** Proven that PSO converges 26% faster with higher fitness ($0.8256$) than GA.

3. **Real-World GIS Data Ingestion & Data Fusion:**
   - **NASA SRTM 30m DEM:** Geodesic metric slope calculation ($dy \approx 389m, dx \approx 303m$) preventing pixel-spacing distortion.
   - **Demographic Demand Fusion:** WorldPop 100m macro population fused with **63,860 real OSM building footprints** (60/40 ratio).
   - **Road Network:** 11,116 OpenStreetMap road segments.
   - **Riparian Flood Buffers:** 749 waterways (65 rivers with 60m buffers, 113 canals with 30m buffers, 272 streams).
   - **Heritage & Protected Buffers:** 125 religious/cultural sites (30m buffer) and Knuckles/Ritigala conservation reserves.

4. **Strict Statutory Hard Constraint Masking:**
   - Automated death penalty ($-999.0$) applied to flood zones, slopes $>20^\circ$, and protected areas.
   - 100% explainable legal reasoning citing Sri Lanka Irrigation Department, NBRO, and UDA regulations.

5. **Reverse HBU Siting & Building Spec Predictor:**
   - Maximum allowable floor count based on NBRO slope categories.
   - Solar PV generation potential (kWp) derived from rooftop orientation and solar irradiance.
   - Construction cost and total budget estimation based on facility capacity.

6. **Interactive 2D GIS & 3D WebGL Decision Dashboard:**
   - **2D:** Leaflet.js map with layers, waterway vectors, and click-to-evaluate reverse siting.
   - **3D:** Three.js WebGL terrain rendering for visual mountain slope analysis.

7. **Ground-Truth Statistical Validation:**
   - Validated against 68 existing schools and 27 existing hospitals in Matale District.
   - **Hospital ROC-AUC:** 88.24%
   - **School ROC-AUC:** 79.20%
   - **Mann-Whitney U Test & Two-Sample t-test:** $p < 0.001$ (statistically significant distinction from random non-facility locations).

---

## 🛠️ Tech Stack & Requirements

- **Programming Language:** Python 3.10 - 3.14
- **Core Scientific Libraries:**
  - `NumPy` & `SciPy` (Matrix mathematics, Eigenvalues, statistical hypothesis testing)
  - `GeoPandas` & `Shapely` (Vector GIS, spatial joins, geodesic buffers)
  - `Rasterio` (GeoTIFF raster processing for DEM and WorldPop grids)
  - `Matplotlib` & `Pillow` (Plotting ROC curves, convergence histories, spatial heatmaps)
  - `scikit-learn` (Metrics, ROC-AUC calculation)
- **Frontend / Visualization:**
  - `HTML5`, `CSS3` (Modern Glassmorphism dark UI)
  - `Vanilla JavaScript (ES6+)` (Asynchronous REST API polling via `fetch()`)
  - `Leaflet.js` (2D GIS vector and tile layers)
  - `Three.js` + `WebGL` (3D terrain mesh generation and interactive orbit controls)

---

## 📁 Repository Directory Structure

```text
spatial_optimization/
├── ahp_engine.py                  # AHP Pairwise comparison & Saaty CR validation
├── building_feature_predictor.py  # NBRO floor limits, solar PV, and budget estimation
├── data_generator.py              # Synthetic/fallback GIS spatial generator
├── download_enhanced_online_data.py # Automated OSM data fetcher
├── download_real_dem.py           # NASA SRTM elevation data downloader
├── evaluate_system.py             # Master benchmark pipeline (Stages 1-4)
├── facility_validation.py         # Ground-truth ROC-AUC & Mann-Whitney testing
├── genetic_algorithm.py           # Spatial GA optimizer implementation
├── pso_optimizer.py               # Spatial PSO swarm optimizer implementation
├── process_enhanced_datasets.py   # Preprocessing OSM waterways & POFW buffers
├── real_data_loader.py            # High-performance multi-source GIS data ingestion
├── requirements.txt               # Python package dependencies
├── sensitivity_analysis.py        # One-At-a-Time (OAT) weight sensitivity analysis
├── statistical_benchmark.py       # Monte Carlo 50-run statistical comparison
├── suitability_surface.py         # WLC 8-criteria suitability surface builder
├── interactive_dashboard.py       # Full-stack HTTP/REST API server & SPA Dashboard
├── data/                          # Small boundary GeoJSONs & utility points
│   ├── matale_boundary.geojson
│   ├── matale_critical_utilities.json
│   └── matale_major_highways.json
├── outputs/                       # Evaluation benchmarks & pre-cached GeoJSONs
│   ├── benchmark_results.json
│   ├── facility_validation.json
│   └── matale_waterways.geojson   # 749 waterway buffer features
└── tests/                         # Automated unit test suite
    ├── test_ahp_engine.py         # Tests for Saaty matrices and CR < 0.10
    ├── test_feature_predictor.py  # Tests for building floor limits & solar math
    └── test_optimizers.py         # Tests for GA/PSO bounds & constraint penalty
```

---

## ⚙️ Installation & Setup Guide

### 1. Clone the Repository
```powershell
git clone https://github.com/ChamodiDilshani/Urban-Growth-Optimizer_IT-337.git
cd Urban-Growth-Optimizer_IT-337
git checkout infrastructure_optimization
```

### 2. Navigate to Component Directory & Install Dependencies
Ensure Python 3.10+ is installed on your system:
```powershell
cd spatial_optimization
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Running the System

### Start the Interactive Decision Dashboard
Execute the server script:
```powershell
python interactive_dashboard.py
```

Once initialized, open your web browser and navigate to:
```
http://localhost:8050
```

### Dashboard Capabilities:
- **Optimization Panel:** Select facility type (School/Hospital), pick a geographic sub-zone, adjust AHP weights, and execute GA vs. PSO.
- **Interactive Siting (Reverse HBU):** Click anywhere on the 2D map to inspect real-time suitability, statutory constraints, slope gradients, and building recommendations.
- **3D Terrain Explorer:** Toggle the 3D view to inspect topographical elevation and hillside terrain in WebGL.

---

## 🧪 Running Automated Unit Tests

Execute the test suite using Python's built-in `unittest` runner:
```powershell
cd spatial_optimization
python -m unittest discover tests -v
```

Expected output:
```text
Ran 12 tests in ~0.1s
OK (All 12 unit tests passing)
```

---

## 📊 Summary of Research Results

| Evaluation Metric | Genetic Algorithm (GA) | Particle Swarm Optimization (PSO) | Superior Optimizer |
|---|---|---|---|
| **Best Fitness Score** | 0.7891 | **0.8256** | **PSO (+4.6%)** |
| **Convergence Speed** | 100 generations | **150 iterations (26% faster)** | **PSO** |
| **Constraint Violation Rate** | 0.0% | 0.0% | **Tie (100% safe)** |
| **Ground-Truth Hospital AUC** | - | **88.24%** | **Statistically Validated** |
| **Ground-Truth School AUC** | - | **79.20%** | **Statistically Validated** |
| **Mann-Whitney U Test** | - | **$p < 0.001$** | **Significantly Better than Random** |

---

## 👥 Contributors & Contact

- **Student:** SENAKA K A S (IT23170902) — *Component 4 Lead*
- **Institution:** Department of Information Technology, SLIIT, Sri Lanka
- **Research Topic:** Multi-Criteria Spatial Optimization for Infrastructure Site Selection
