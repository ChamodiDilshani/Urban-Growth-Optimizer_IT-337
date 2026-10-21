import mongoose from 'mongoose';

let isConnected = false;

export const connectDB = async () => {
  const mongoURI = process.env.MONGO_URI || 'mongodb+srv://mepradeepad_db_user:9zs0fMdMmmwEswRb@cluster0.9dq9zz2.mongodb.net/';

  try {
    mongoose.set('strictQuery', false);
    await mongoose.connect(mongoURI, {
      dbName: 'urbanscapes',
      serverSelectionTimeoutMS: 8000,
    });
    isConnected = true;
    console.log('[MongoDB] Connected successfully to MongoDB Atlas (database: urbanscapes)');
  } catch (error) {
    isConnected = false;
    console.warn(`[MongoDB Warning] Could not connect to MongoDB Atlas: ${error.message}`);
    console.warn('[MongoDB Notice] In the meantime, valuations will be stored in the session cache.');
  }
};

export const isDBConnected = () => isConnected;
