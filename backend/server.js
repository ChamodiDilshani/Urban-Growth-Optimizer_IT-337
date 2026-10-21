import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { connectDB } from './config/db.js';
import valuationRoutes from './routes/valuationRoutes.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Connect to Database
connectDB();

// Middleware
app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// Health Check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Urbanscapes-LK Express API',
    timestamp: new Date().toISOString(),
  });
});

// Valuation API Routes
app.use('/api/valuation', valuationRoutes);

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('Unhandled server error:', err);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: err.message,
  });
});

// Start Express Server with graceful port in-use handling
const server = app.listen(PORT, () => {
  console.log(`[Express] Server running on http://localhost:${PORT}`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.warn(`\n⚠️  [Express Error] Port ${PORT} is already in use by another process.`);
    console.warn(`Attempting to automatically free port ${PORT}...`);
    try {
      import('child_process').then(({ execSync }) => {
        execSync(`powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort ${PORT} -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"`, { stdio: 'ignore' });
        console.log(`[Express] Port ${PORT} freed successfully. Please restart npm start.\n`);
        process.exit(0);
      }).catch(() => {
        console.error(`To free port ${PORT}, run: Stop-Process -Id (Get-NetTCPConnection -LocalPort ${PORT}).OwningProcess -Force\n`);
        process.exit(1);
      });
    } catch {
      process.exit(1);
    }
  } else {
    console.error('[Express Error]', err);
  }
});
