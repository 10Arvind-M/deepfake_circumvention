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

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_NAME_OR_DIR = "google/vit-base-patch16-224-in21k"

# Where to save model after download
SAVED_STATE_PATH = "vit_deepfake_model.pth"

# HuggingFace direct download link (replace with your real link)
# Example format:
# https://huggingface.co/<username>/<repo>/resolve/main/vit_deepfake_model.pth
HUGGINGFACE_MODEL_URL = os.environ.get(
    "HF_MODEL_URL",
    "https://huggingface.co/arvind-1001-m/deepfake_model/resolve/main/vit_deepfake_model.pth"
)

NUM_FRAMES_DEFAULT = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# DOWNLOAD MODEL FROM HUGGINGFACE ON FIRST RUN
# ------------------------------------------------------------
def download_model_if_needed():
    """
    Download the .pth model file from HuggingFace if not already present.
    Works on Render free tier (no disk needed).
    """
    if Path(SAVED_STATE_PATH).exists():
        print("Model already exists, skipping download.")
        return

    print("Downloading model from HuggingFace...")
    try:
        r = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
        r.raise_for_status()

        with open(SAVED_STATE_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Model downloaded successfully:", SAVED_STATE_PATH)

    except Exception as e:
        print("❌ MODEL DOWNLOAD FAILED:", e)
        raise RuntimeError("Could not download model from HuggingFace") from e


# ------------------------------------------------------------
# LOAD MODEL + PROCESSOR
# ------------------------------------------------------------
def load_model_and_processor():
    # Ensure model exists
    download_model_if_needed()

    # Load processor
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME_OR_DIR)

    # Create model
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME_OR_DIR,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()

    # Load trained weights
    try:
        state = torch.load(SAVED_STATE_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Loaded model weights from:", SAVED_STATE_PATH)
    except Exception as e:
        print("❌ Failed to load state_dict:", e)

    return model, processor


MODEL, PROCESSOR = load_model_and_processor()


# ------------------------------------------------------------
# FRAME EXTRACTION FOR VIDEO
# ------------------------------------------------------------
def extract_frames(video_path: str, num_frames: int = NUM_FRAMES_DEFAULT):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count == 0:
        cap.release()
        return []

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


# ------------------------------------------------------------
# PREDICTION FOR FRAMES
# ------------------------------------------------------------
def predict_frames(frames):
    if len(frames) == 0:
        raise ValueError("No frames provided")

    enc = PROCESSOR(images=frames, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(device)

    with torch.no_grad():
        outputs = MODEL(pixel_values=pixel_values).logits
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    mean_probs = probs.mean(axis=0)
    return {"real": float(mean_probs[0]), "fake": float(mean_probs[1])}, probs


# ------------------------------------------------------------
# ROUTE: PREDICT IMAGE
# ------------------------------------------------------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        img_path = tmp.name

    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    mean_probs, frame_probs = predict_frames([img])
    predicted = "fake" if mean_probs["fake"] > mean_probs["real"] else "real"

    return JSONResponse({
        "predicted_label": predicted,
        "mean_probabilities": mean_probs,
        "per_frame_probs": frame_probs.tolist()
    })


# ------------------------------------------------------------
# ROUTE: PREDICT VIDEO
# ------------------------------------------------------------
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...), num_frames: int = NUM_FRAMES_DEFAULT):
    suffix = Path(file.filename).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        video_path = tmp.name

    frames = extract_frames(video_path, num_frames=int(num_frames))
    if len(frames) == 0:
        raise HTTPException(status_code=400, detail="Could not extract frames")

    mean_probs, frame_probs = predict_frames(frames)
    predicted = "fake" if mean_probs["fake"] > mean_probs["real"] else "real"
    per_frame_conf = [float(p.max()) for p in frame_probs]

    return JSONResponse({
        "predicted_label": predicted,
        "mean_probabilities": mean_probs,
        "per_frame_probs": frame_probs.tolist(),
        "per_frame_max_confidence": per_frame_conf
    })


# ------------------------------------------------------------
# RUN SERVER LOCALLY / IN DOCKER
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000)
