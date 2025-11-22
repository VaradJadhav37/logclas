import streamlit as st
import pandas as pd
import io
import os
import subprocess
import sys
import time
import traceback
import requests

st.set_page_config(page_title="Log Classifier (Streamlit + FastAPI)", layout="wide")
st.title("Hybrid Log Classification — Regex + BERT + LLM Fallback")

st.markdown("""
This UI integrates with your **FastAPI backend**.

### How it works:
- You upload a CSV containing **source** and **log_message**
- Streamlit POSTs the file to your FastAPI `/classify/` endpoint
- The backend returns a classified CSV
- You download the result

### Note:
This app does NOT import backend classify code — no TensorFlow DLL issues.
""")

# ------------------------------------------------------------------------------
# FASTAPI SERVER CONTROL
# ------------------------------------------------------------------------------

FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
FASTAPI_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/classify/"

st.subheader("Backend Server")

if "uvicorn_proc" not in st.session_state:
    st.session_state.uvicorn_proc = None

col1, col2 = st.columns(2)

with col1:
    if st.button("Start FastAPI Server"):
        if st.session_state.uvicorn_proc is None:
            cmd = [
                sys.executable, "-m", "uvicorn",
                "server:app",
                "--host", FASTAPI_HOST,
                "--port", str(FASTAPI_PORT),
                "--reload"
            ]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.session_state.uvicorn_proc = proc
                st.success("FastAPI server started.")
                time.sleep(1)
            except Exception as e:
                st.error(f"Failed to start FastAPI: {e}")
                st.text(traceback.format_exc())
        else:
            st.info("Server already running (started by this UI).")

with col2:
    if st.button("Stop FastAPI Server"):
        if st.session_state.uvicorn_proc:
            st.session_state.uvicorn_proc.terminate()
            st.session_state.uvicorn_proc.wait(timeout=5)
            st.session_state.uvicorn_proc = None
            st.success("FastAPI server stopped.")
        else:
            st.info("No server started via this UI.")

st.write(f"**FastAPI Endpoint:** `{FASTAPI_URL}`")
st.markdown("---")

# ------------------------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload CSV (must include source & log_message columns)", type=["csv"])

def post_to_fastapi(csv_bytes):
    files = {"file": ("upload.csv", csv_bytes, "text/csv")}
    resp = requests.post(FASTAPI_URL, files=files, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"Error {resp.status_code}: {resp.text}")

    return pd.read_csv(io.BytesIO(resp.content))

# ------------------------------------------------------------------------------
# RUN CLASSIFICATION
# ------------------------------------------------------------------------------

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
else:
    df = pd.read_csv(uploaded_file)
    
    if "source" not in df.columns or "log_message" not in df.columns:
        st.error("CSV must contain columns: 'source', 'log_message'")
    else:
        st.subheader("Preview")
        st.dataframe(df.head())

        if st.button("Run Classification"):
            try:
                with st.spinner("Sending file to FastAPI..."):
                    csv_bytes = uploaded_file.getvalue()
                    out_df = post_to_fastapi(csv_bytes)
                st.success("Classification complete.")
                st.dataframe(out_df.head())

                st.download_button(
                    "Download Classified CSV",
                    out_df.to_csv(index=False).encode("utf-8"),
                    "classified_output.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Classification failed: {e}")
                st.text(traceback.format_exc())
