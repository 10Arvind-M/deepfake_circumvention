# inference_server.py
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import requests
import numpy as np
import torch
import cv2
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI(title="ViT Deepfake Inference API")

# Config (environment override)
MODEL_NAME_OR_DIR = os.environ.get("MODEL_NAME_OR_DIR", "google/vit-base-patch16-224-in21k")
SAVED_STATE_PATH = "/var/data/vit_deepfake_model.pth"
SAVED_PRETRAINED_DIR = os.environ.get("SAVED_PRETRAINED_DIR", "saved_model")
MODEL_DOWNLOAD_URL = os.environ.get("MODEL_DOWNLOAD_URL", None)  # presigned S3 URL
NUM_FRAMES_DEFAULT = int(os.environ.get("NUM_FRAMES_DEFAULT", "5"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def maybe_download_model(download_url: str, dest_path: str):
    if not download_url:
        return False, "No download URL provided"
    try:
        r = requests.get(download_url, stream=True, timeout=600)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)

def load_model_and_processor():
    # Attempt to download if URL provided and file not present
    if MODEL_DOWNLOAD_URL:
        dest = Path(SAVED_STATE_PATH)
        if not dest.exists():
            ok, info = maybe_download_model(MODEL_DOWNLOAD_URL, str(dest))
            if not ok:
                print("Model download failed:", info)
            else:
                print("Model downloaded to", dest)

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME_OR_DIR)

    if Path(SAVED_PRETRAINED_DIR).exists():
        model = ViTForImageClassification.from_pretrained(SAVED_PRETRAINED_DIR)
        model.to(device)
        model.eval()
        print("Loaded model from saved pretrained directory:", SAVED_PRETRAINED_DIR)
        return model, processor

    model = ViTForImageClassification.from_pretrained(MODEL_NAME_OR_DIR, num_labels=2, ignore_mismatched_sizes=True)
    model.to(device)
    model.eval()

    if Path(SAVED_STATE_PATH).exists():
        try:
            state = torch.load(SAVED_STATE_PATH, map_location=device)
            model.load_state_dict(state, strict=False)
            print("Loaded state_dict from", SAVED_STATE_PATH)
        except Exception as e:
            print("Warning: failed to load state_dict:", e)
    else:
        print("No saved state_dict found; model will use pretrained backbone + new head (not trained).")

    return model, processor

MODEL, PROCESSOR = load_model_and_processor()

def extract_frames(video_path: str, num_frames: int = NUM_FRAMES_DEFAULT):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count == 0:
        cap.release()
        return frames
    step = max(1, frame_count // num_frames)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames

def predict_frames(frames):
    if len(frames) == 0:
        raise ValueError("No frames provided for prediction")
    enc = PROCESSOR(images=frames, return_tensors="pt")
    pixel_values = enc['pixel_values'].to(device)
    with torch.no_grad():
        outputs = MODEL(pixel_values=pixel_values).logits
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    mean_probs = probs.mean(axis=0)
    return {"real": float(mean_probs[0]), "fake": float(mean_probs[1])}, probs

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        tmp_path = tmp.name
    img = cv2.imread(tmp_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    mean_probs, per_frame_probs = predict_frames([img])
    predicted = "fake" if mean_probs["fake"] > mean_probs["real"] else "real"
    return JSONResponse({
        "predicted_label": predicted,
        "mean_probabilities": mean_probs,
        "per_frame_probs": per_frame_probs.tolist()
    })

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...), num_frames: int = NUM_FRAMES_DEFAULT):
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        tmp_path = tmp.name
    frames = extract_frames(tmp_path, num_frames=int(num_frames))
    if len(frames) == 0:
        raise HTTPException(status_code=400, detail="Could not extract frames from uploaded video")
    mean_probs, per_frame_probs = predict_frames(frames)
    predicted = "fake" if mean_probs["fake"] > mean_probs["real"] else "real"
    per_frame_max_conf = [float(p.max()) for p in per_frame_probs]
    return JSONResponse({
        "predicted_label": predicted,
        "mean_probabilities": mean_probs,
        "per_frame_probs": per_frame_probs.tolist(),
        "per_frame_max_confidence": per_frame_max_conf
    })

if __name__ == "__main__":
    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000, log_level="info")
