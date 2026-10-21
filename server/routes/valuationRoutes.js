import express from 'express';
import axios from 'axios';
import Valuation from '../models/Valuation.js';
import { isDBConnected } from '../config/db.js';

const router = express.Router();
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:5001';

// In-memory fallback history if MongoDB is not connected
const inMemoryHistory = [];

// 1. Fetch metadata (suburbs, zoning categories, road widths, coordinates)
router.get('/metadata', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_SERVICE_URL}/metadata`, { timeout: 15000 });
    res.json({
      success: true,
      data: response.data,
      dbConnected: isDBConnected(),
    });
  } catch (error) {
    console.error('Error fetching metadata from Python service:', error.message);
    res.status(502).json({
      success: false,
      message: 'Failed to communicate with Python Model Engine. Ensure the Python service is running.',
      error: error.message,
    });
  }
});

// 2. Perform Valuation Prediction + SHAP Generation
router.post('/predict', async (req, res) => {
  try {
    const { suburb, land_size, land_type, road_width, target_year, lat, lon } = req.body;

    if (!suburb) {
      return res.status(400).json({ success: false, message: 'suburb is required' });
    }

    // Call Python FastAPI service
    const response = await axios.post(`${PYTHON_SERVICE_URL}/predict`, {
      suburb,
      land_size: Number(land_size) || 12.0,
      land_type: land_type || 'Residential',
      road_width: road_width || '15 ft',
      target_year: Number(target_year) || 2026,
      lat: Number(lat) || 6.9022,
      lon: Number(lon) || 79.9172,
    }, { timeout: 30000 });

    const result = response.data;

    // Save record to MongoDB or in-memory fallback
    const recordData = {
      suburb: result.suburb,
      landSize: result.land_size,
      landType: result.land_type,
      roadWidth: result.road_width,
      targetYear: result.target_year,
      coordinates: result.coordinates,
      distFort: result.dist_fort,
      isFloodRisk: result.is_flood_risk,
      finalPricePerPerch: result.final_price_per_perch,
      totalValuation: result.total_valuation,
      annualAssessedValue: result.annual_assessed_value,
      annualTax: result.annual_tax,
      taxRate: result.tax_rate,
      yieldRate: result.yield_rate,
      multiYearTax: result.multi_year_tax,
    };

    let savedId = null;
    if (isDBConnected()) {
      try {
        const doc = await Valuation.create(recordData);
        savedId = doc._id;
      } catch (dbErr) {
        console.warn('Failed to save to MongoDB:', dbErr.message);
      }
    } else {
      const fallbackDoc = {
        _id: 'mem_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
        ...recordData,
        createdAt: new Date().toISOString(),
      };
      inMemoryHistory.unshift(fallbackDoc);
      if (inMemoryHistory.length > 50) inMemoryHistory.pop();
      savedId = fallbackDoc._id;
    }

    res.json({
      success: true,
      data: {
        ...result,
        recordId: savedId,
      },
      dbConnected: isDBConnected(),
    });
  } catch (error) {
    console.error('Error in /predict:', error.response?.data || error.message);
    res.status(500).json({
      success: false,
      message: 'Valuation prediction failed',
      detail: error.response?.data?.detail || error.message,
    });
  }
});

// 3. Generate PDF Report (returns base64 JSON to avoid IDM browser interception)
router.post('/pdf', async (req, res) => {
  try {
    const { parcel_info, valuation_info } = req.body;

    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/generate-pdf`,
      { parcel_info, valuation_info },
      { timeout: 20000 }
    );

    res.json({
      success: true,
      pdf_base64: response.data.pdf_base64,
    });
  } catch (error) {
    console.error('Error generating PDF:', error.message);
    res.status(500).json({
      success: false,
      message: 'Failed to generate PDF report',
      detail: error.message,
    });
  }
});

// 4. Retrieve Valuation History (MongoDB or in-memory)
router.get('/history', async (req, res) => {
  try {
    if (isDBConnected()) {
      const records = await Valuation.find().sort({ createdAt: -1 }).limit(25).lean();
      return res.json({ success: true, count: records.length, data: records, source: 'mongodb' });
    } else {
      return res.json({ success: true, count: inMemoryHistory.length, data: inMemoryHistory, source: 'memory' });
    }
  } catch (error) {
    console.error('Error fetching history:', error.message);
    res.status(500).json({ success: false, message: 'Could not fetch history' });
  }
});

// 5. Delete specific history record
router.delete('/history/:id', async (req, res) => {
  try {
    const { id } = req.params;
    if (isDBConnected()) {
      await Valuation.findByIdAndDelete(id);
    } else {
      const idx = inMemoryHistory.findIndex((item) => item._id === id);
      if (idx !== -1) inMemoryHistory.splice(idx, 1);
    }
    res.json({ success: true, message: 'Record deleted successfully' });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});

// 6. Clear all history
router.delete('/history', async (req, res) => {
  try {
    if (isDBConnected()) {
      await Valuation.deleteMany({});
    }
    inMemoryHistory.length = 0;
    res.json({ success: true, message: 'Valuation history cleared' });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});

export default router;
