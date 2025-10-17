# Skin Cancer Prediction Web App

## Structure
- frontend/ (Vite React app)
- backend/ (Express + MongoDB)
- ml_service/ (FastAPI + PyTorch)
- database/ (Mongo volume)

## Local Development (Windows PowerShell)
1. Start MongoDB locally on 27017, or use Docker Compose below.
2. ML Service
   ```bash
   cd ml_service
   python -m venv .venv
   .venv\Scripts\Activate
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```
3. Backend
   ```bash
   cd backend
   npm install
   copy .env.example .env
   npm run dev
   ```
4. Frontend
   ```bash
   cd frontend
   npm install
   set VITE_BACKEND_URL=http://localhost:8000
   npm run dev
   ```

## Docker Compose (recommended)
```bash
docker compose up --build
```
- Frontend: http://localhost:5173
- Backend: http://localhost:8000/health
- ML Service: http://localhost:8001/health

## Notes
- Simple `SimpleCNN` 2-class head (demo purposes; not for clinical use).
- Heatmap produced via a Grad-CAM-like visualization for interpretability.

## Training with HAM10000 metadata

This repo includes a training script that uses the HAM10000 metadata CSV to map `image_id` to labels and trains a small CNN. Place the HAM images in a folder (e.g., `data/ham_images`) and ensure the metadata CSV exists (e.g., `data/train/HAM10000_metadata.csv`).

Example (Windows PowerShell):
```bash
.\.venv\Scripts\python.exe ml_service\train_ham.py \
  --metadata-csv data\train\HAM10000_metadata.csv \
  --images-dir data\ham_images \
  --epochs 8 --batch-size 16 --lr 1e-3 \
  --out ml_service\model_weights.pth
```

- The ML service automatically loads `ml_service/model_weights.pth` if present.
- If weights are not found, the service runs with randomly initialized weights (for development only).

### Notes on data usage
- This project does not bundle any third-party images. You must provide the HAM images yourself to train.
- The app can still accept photos for inference. For meaningful predictions, train and save weights first.

### Sample testing
- After training and starting all services, upload a lesion photo in the UI and optionally provide symptoms; the backend forwards the image and symptom JSON to the ML service, which fuses both into the final prediction.