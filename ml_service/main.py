from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import os
import torch
import torch.nn as nn
import numpy as np
import base64
import cv2
import json

app = FastAPI(title='Skin Lesion ML Service')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # 3x224x224 -> feature maps
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 x 112 x 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 28 x 28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits, feats


model = SimpleCNN(num_classes=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
# Try load trained weights if available
try:
    # Resolve path robustly (works when cwd is repo root or service dir)
    candidate_paths = [
        os.path.join(os.getcwd(), 'ml_service', 'model_weights.pth'),
        os.path.join(os.getcwd(), 'model_weights.pth'),
        os.path.join(os.path.dirname(__file__), 'model_weights.pth'),
    ]
    weight_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if weight_path:
        state = torch.load(weight_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif isinstance(state, dict):
            model.load_state_dict(state)
    model.eval()
except Exception:
    # Fallback to randomly initialized eval model
    model.eval()

# Probability temperature (sharpening). Lower than 1.0 spreads predictions toward 0%/100%.
TEMPERATURE = float(os.environ.get('ML_TEMPERATURE', '0.7'))

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    # Normalize similar to ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr)
    return tensor

idx_to_class = {0: 'benign', 1: 'malignant'}


class GradCAM:
    def __init__(self, model: SimpleCNN):
        self.model = model

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        # Ensure gradient is tracked only for this scope
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.requires_grad_(True)
        # Retain gradients on the last conv features by re-running forward and keeping ref
        logits, feats = self.model(input_tensor)
        feats.retain_grad()
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward()
        grads = feats.grad
        if grads is None:
            # Should not happen, but guard anyway
            return np.zeros((224, 224), dtype=np.float32)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feats).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam


cam_generator = GradCAM(model)


@app.get('/health')
async def health():
    return {'status': 'ok'}


def _symptom_risk_from_dict(symptoms: dict) -> tuple[float, dict]:
    # Normalize keys to lowercase
    normalized = {}
    for k, v in (symptoms or {}).items():
        try:
            key = str(k).strip().lower()
            normalized[key] = v
        except Exception:
            continue

    # Interpret booleans/strings/numbers
    def as_bool(x):
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        if isinstance(x, (int, float)):
            return 1.0 if float(x) > 0 else 0.0
        if isinstance(x, str):
            s = x.strip().lower()
            return 1.0 if s in { '1', 'true', 'yes', 'y' } else 0.0
        return 0.0

    def as_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    # Features with simple weights (toy prior; for demo purposes only)
    features = {
        'itching': as_bool(normalized.get('itching')),
        'bleeding': as_bool(normalized.get('bleeding')),
        'pain': as_bool(normalized.get('pain')),
        'color_change': as_bool(normalized.get('color_change')) or as_bool(normalized.get('colour_change')),
        'rapid_growth': as_bool(normalized.get('rapid_growth')) or as_bool(normalized.get('growing')),
        'border_irregularity': as_bool(normalized.get('border_irregularity')) or as_bool(normalized.get('irregular_border')),
        'diameter_mm': min(max(as_float(normalized.get('diameter_mm', 0.0)), 0.0), 100.0),
        'age': min(max(as_float(normalized.get('age', 0.0)), 0.0), 120.0),
        'family_history': as_bool(normalized.get('family_history')),
    }

    # Weights roughly reflecting higher risk indicators
    w = {
        'itching': 0.2,
        'bleeding': 0.5,
        'pain': 0.2,
        'color_change': 0.6,
        'rapid_growth': 0.7,
        'border_irregularity': 0.7,
        'diameter_mm': 0.03,  # each mm adds risk up to a cap
        'age': 0.005,         # small increase with age
        'family_history': 0.5,
    }

    # Linear combination then squashed into [0,1]
    linear = sum(features[k] * w[k] for k in w)
    # cap the linear value to avoid extreme probabilities
    linear = max(0.0, min(linear, 4.0))
    risk = 1.0 / (1.0 + np.exp(-(linear - 2.0)))  # center near 2.0

    return float(risk), features


@app.post('/predict')
async def predict(file: UploadFile = File(...), symptoms: str | None = Form(None)):
    try:
        bytes_data = await file.read()
        img = Image.open(io.BytesIO(bytes_data)).convert('RGB')
        input_tensor = preprocess_image(img).unsqueeze(0).to(device)
        # Simple OOD guard: require sufficient skin-like pixels to proceed
        # Compute on resized numpy image in HSV space
        img_resized = np.array(img.resize((224, 224)))
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        # Broad skin range (heuristic)
        lower = np.array([0, 20, 50], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower, upper)
        lower2 = np.array([160, 20, 50], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        skin_ratio = float((skin_mask > 0).mean())
        # Additional heuristics for non-photographic/unrelated images
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray_std = float(gray.std())
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        sat_mean = float(sat.mean())
        # Flag OOD if very little skin region OR extremely low texture (icons, drawings)
        ood_flag = (skin_ratio < 0.15) or (gray_std < 18.0)
        input_tensor.requires_grad_(True)
        logits, _ = model(input_tensor)
        # Temperature-scaled softmax for wider spread [0,1]
        probs_tensor = torch.softmax(logits / TEMPERATURE, dim=1)[0]
        probs = probs_tensor.detach().cpu().numpy().tolist()
        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_class[pred_idx]
        confidence = float(np.max(probs))

        # Parse and incorporate symptoms if provided
        parsed_symptoms = None
        symptom_features = None
        symptom_risk = None
        if symptoms:
            try:
                parsed_symptoms = json.loads(symptoms)
            except Exception:
                parsed_symptoms = {'raw': symptoms}
            symptom_risk, symptom_features = _symptom_risk_from_dict(parsed_symptoms if isinstance(parsed_symptoms, dict) else {})

            # Blend malignant probability with symptom risk (simple late fusion)
            p_benign = float(probs[0])
            p_malignant = float(probs[1])
            alpha = 0.15  # lower weight on symptom prior to reduce bias
            p_malignant_fused = (1.0 - alpha) * p_malignant + alpha * float(symptom_risk)
            p_benign_fused = 1.0 - p_malignant_fused
            probs = [p_benign_fused, p_malignant_fused]
            pred_idx = int(np.argmax(probs))
            pred_label = idx_to_class[pred_idx]
            confidence = float(max(probs))

        # Apply conservative thresholding and OOD rejection
        malignant_prob = float(probs[1])
        threshold = 0.60
        if ood_flag:
            pred_label = 'benign'
            malignant_prob = 0.0
            probs = [1.0, 0.0]
            # Force displayed confidence to 0% for unrelated images
            confidence = 0.0
        elif malignant_prob < threshold:
            pred_label = 'benign'
            probs = [1.0 - malignant_prob, malignant_prob]
            confidence = float(max(probs))

        cam = cam_generator.generate(input_tensor, class_idx=pred_idx)
        img_np = np.array(img.resize((224, 224)))
        heatmap = (cam * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        _, buf = cv2.imencode('.jpg', overlay)
        heatmap_b64 = base64.b64encode(buf).decode('utf-8')

        response = {
            'prediction': pred_label,
            'confidence': confidence,
            'probs': {idx_to_class[i]: float(p) for i, p in enumerate(probs)},
            'heatmap_base64': 'data:image/jpeg;base64,' + heatmap_b64,
        }
        response['meta'] = {
            'skin_ratio': skin_ratio,
            'gray_std': gray_std,
            'sat_mean': sat_mean,
            'threshold': threshold,
            'ood_rejected': bool(ood_flag)
        }
        if ood_flag:
            response['heatmap_base64'] = None
        if symptoms is not None:
            response['symptoms'] = parsed_symptoms
            if symptom_risk is not None:
                # For unrelated images, force symptom_risk display to 0%
                response['symptom_risk'] = 0.0 if ood_flag else float(symptom_risk)
                response['symptom_features'] = symptom_features
        elif ood_flag:
            # Ensure UI shows 0% prior even if no symptoms were provided
            response['symptom_risk'] = 0.0
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)


