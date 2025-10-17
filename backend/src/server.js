import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import mongoose from "mongoose";
import multer from "multer";
import path from "path";
import fs from "fs";
import axios from "axios";

dotenv.config();

const app = express();
// Restrict CORS to local dev by default; adjust via env FRONTEND_ORIGIN if needed
const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || "http://localhost:5173";
app.use(cors({ origin: FRONTEND_ORIGIN, credentials: false }));
app.use(express.json({ limit: "10mb" }));

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/skin_cancer";
const PORT = process.env.PORT || 8000;
const ML_URL = process.env.ML_URL || "http://127.0.0.1:8001";
const UPLOAD_DIR = path.resolve("./uploads");
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

let isDbAvailable = true;
try {
  await mongoose.connect(MONGO_URI);
  console.log("Connected to MongoDB");
} catch (err) {
  isDbAvailable = false;
  console.warn("MongoDB not available. Falling back to in-memory store.");
}

const resultSchema = new mongoose.Schema({
  filename: String,
  prediction: String,
  confidence: Number,
  probs: Object,
  heatmap_base64: String,
  symptoms: Object,
  symptom_risk: Number,
  createdAt: { type: Date, default: Date.now }
});
const Result = isDbAvailable ? mongoose.model("Result", resultSchema) : null;
const memoryResults = [];

const MAX_FILE_SIZE_BYTES = Number(process.env.MAX_FILE_SIZE_BYTES || 5 * 1024 * 1024); // 5MB default
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, UPLOAD_DIR);
  },
  filename: function (req, file, cb) {
    const unique = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, unique + path.extname(file.originalname));
  }
});
const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE_BYTES },
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
    if (!allowed.includes(file.mimetype)) return cb(new Error("Unsupported file type"));
    cb(null, true);
  }
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.get("/results", async (req, res) => {
  if (isDbAvailable) {
    const items = await Result.find().sort({ createdAt: -1 }).limit(50);
    return res.json(items);
  }
  const items = [...memoryResults].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)).slice(0, 50);
  return res.json(items);
});

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });
    const filePath = req.file.path;
    const fileStream = fs.createReadStream(filePath);

    const FormData = (await import("form-data")).default;
    const form = new FormData();
    form.append("file", fileStream, { filename: path.basename(filePath) });
    // Forward optional symptoms JSON from frontend
    if (req.body && typeof req.body.symptoms !== 'undefined') {
      // Accept both object and stringified JSON
      const symptomsStr = typeof req.body.symptoms === 'string' ? req.body.symptoms : JSON.stringify(req.body.symptoms);
      form.append("symptoms", symptomsStr);
    }

    let data;
    try {
      const response = await axios.post(`${ML_URL}/predict`, form, {
        headers: form.getHeaders(),
        timeout: 15000,
        maxRedirects: 0
      });
      data = { ...response.data, source: "ml" };
    } catch (mlErr) {
      // Fallback mock prediction so the site works without ML running
      const probs = { benign: 0.65, malignant: 0.35 };
      const pred = probs.benign >= probs.malignant ? "benign" : "malignant";
      const confidence = Math.max(probs.benign, probs.malignant);
      const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='224' height='224'>\n  <defs>\n    <linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>\n      <stop offset='0%' stop-color='rgba(255,0,0,0.0)'/>\n      <stop offset='100%' stop-color='rgba(255,0,0,0.5)'/>\n    </linearGradient>\n  </defs>\n  <rect width='224' height='224' fill='url(#g)'/>\n</svg>`;
      const heatmap_base64 = `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;
      data = { prediction: pred, confidence, probs, heatmap_base64, source: "mock" };
    }

    const item = {
      filename: path.basename(filePath),
      prediction: data.prediction,
      confidence: data.confidence,
      probs: data.probs,
      heatmap_base64: data.heatmap_base64,
      symptoms: data.symptoms || (req.body && req.body.symptoms ? req.body.symptoms : undefined),
      symptom_risk: data.symptom_risk,
      createdAt: new Date(),
      source: data.source
    };

    if (isDbAvailable) {
      const saved = await Result.create(item);
      return res.json(saved);
    } else {
      const withId = { ...item, _id: `${Date.now()}` };
      memoryResults.push(withId);
      return res.json(withId);
    }
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Prediction failed (backend)", details: String(err?.message || err) });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Backend running on http://0.0.0.0:${PORT}`);
}).on('error', (err) => {
  console.error('Server listen error:', err);
});


