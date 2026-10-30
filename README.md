# Urban Growth Optimizer

An intelligent urban planning and development system designed for the Urban Development Authority of Sri Lanka.

## Overview

This project focuses on using data-driven solutions to improve urban development decisions through:

- Spatial growth prediction
- Unauthorized building detection
- Automated land valuation
- Infrastructure optimization

## Project Goals

The platform aims to support smarter planning, reduce unauthorized construction, improve resource allocation, and help authorities make faster and more accurate decisions related to land use and city expansion.

## Features

- Predict future urban growth patterns using spatial and geospatial data
- Detect unauthorized or non-compliant buildings
- Estimate land value more efficiently and consistently
- Optimize infrastructure planning based on growth trends
- Provide a scalable backend and user-friendly frontend interface

## Tech Stack

### Backend
- Python
- FastAPI
- NumPy
- OpenCV
- Shapely
- PyTorch

### Frontend
- React
- Vite
- Tailwind CSS

## Project Structure

```text
Urban-Growth-Optimizer_IT-337/
├── backend/
│   ├── app/
│   ├── models/
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
├── README.md
└── .gitignore
```

## Getting Started

### 1. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 2. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

## Usage

Run the backend API and frontend application together to access the system and test the urban planning workflows.

## Status

This project is currently under development.

## License

This project is intended for academic and research use within the scope of the Urban Development Authority of Sri Lanka.

