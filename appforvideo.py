# streamlit_frontend_api.py
import streamlit as st
import tempfile
from pathlib import Path
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Deepfake Detection (Frontend -> API)", layout="centered")
st.title("Deepfake Detection — Frontend (calls FastAPI)")

try:
    default_api = st.secrets["API_URL"]
except:
    default_api = "http://localhost:8000"

api_url = st.sidebar.text_input("API URL", value=default_api)
num_frames = st.sidebar.slider("Frames to sample (server-side)", min_value=1, max_value=12, value=5)
mode = st.radio("Mode", ["Image", "Video"])

file_types = ["jpg", "jpeg", "png"] if mode == "Image" else ["mp4", "mov", "avi"]
uploaded = st.file_uploader("Upload file", type=file_types)

def save_tmp_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    if suffix == "":
        suffix = ".mp4" if mode=="Video" else ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

if uploaded is None:
    st.info("Upload a file to start.")
    st.stop()

local_path = save_tmp_upload(uploaded)

if mode == "Video":
    # Show the video using the temporary path
    try:
        st.video(local_path)
    except Exception as e:
        st.warning("Streamlit could not render video preview (codec issue). You can still upload and predict.")
else:
    st.image(local_path, use_column_width=True)

if st.button("Predict (call API)"):
    endpoint = "/predict_image" if mode == "Image" else "/predict_video"
    url = api_url.rstrip("/") + endpoint
    files = {"file": open(local_path, "rb")}
    data = {"num_frames": num_frames} if mode == "Video" else {}
    try:
        with st.spinner("Calling inference API..."):
            resp = requests.post(url, files=files, data=data, timeout=120)
            resp.raise_for_status()
            j = resp.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    st.success(f"Predicted: {j.get('predicted_label')}")
    st.write("Mean probabilities:", j.get("mean_probabilities"))
    per_frame = j.get("per_frame_probs")
    if per_frame:
        df = pd.DataFrame(per_frame, columns=["prob_real","prob_fake"])
        df["frame_index"] = df.index
        st.subheader("Per-frame probabilities")
        st.dataframe(df)
        # show per-frame max confidence
        if "per_frame_max_confidence" in j:
            st.write("Per-frame max confidence:", j["per_frame_max_confidence"])
