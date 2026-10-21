import mongoose from 'mongoose';

const valuationSchema = new mongoose.Schema(
  {
    suburb: { type: String, required: true },
    landSize: { type: Number, required: true },
    landType: { type: String, required: true },
    roadWidth: { type: String, required: true },
    targetYear: { type: Number, required: true },
    coordinates: {
      lat: { type: Number, required: true },
      lon: { type: Number, required: true },
    },
    distFort: { type: Number, required: true },
    isFloodRisk: { type: Boolean, default: false },
    finalPricePerPerch: { type: Number, required: true },
    totalValuation: { type: Number, required: true },
    annualAssessedValue: { type: Number, required: true },
    annualTax: { type: Number, required: true },
    taxRate: { type: Number, required: true },
    yieldRate: { type: Number, required: true },
    multiYearTax: { type: Map, of: Number },
  },
  {
    timestamps: true,
  }
);

const Valuation = mongoose.model('Valuation', valuationSchema);

export default Valuation;
