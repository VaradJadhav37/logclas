# server.py (safer FastAPI endpoint)
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import io
import traceback

app = FastAPI()
@app.get("/")
def read_root():
    return {"status": "Log Classifier FastAPI is running."}

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    try:
        # Read the uploaded CSV into memory
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        # Import classify lazily to avoid heavy imports on server startup
        try:
            from classify import classify
        except Exception as e:
            # Return a descriptive error instead of crashing the server
            tb = traceback.format_exc()
            return JSONResponse(status_code=500, content={
                "error": "Failed to import classify() in backend.",
                "detail": str(e),
                "traceback": tb
            })

        # Perform classification (this may still raise runtime errors if models fail)
        try:
            inputs = list(zip(df["source"].astype(str).tolist(), df["log_message"].astype(str).tolist()))
            labels = classify(inputs)
            if labels is None or len(labels) != len(inputs):
                raise RuntimeError("classify() returned invalid result.")
            df["target_label"] = labels
        except Exception as e:
            tb = traceback.format_exc()
            return JSONResponse(status_code=500, content={
                "error": "Classification failed at runtime.",
                "detail": str(e),
                "traceback": tb
            })

        # Save output to resources/ and return file
        os.makedirs("resources", exist_ok=True)
        output_file = os.path.join("resources", "output.csv")
        df.to_csv(output_file, index=False)
        return FileResponse(output_file, media_type='text/csv', filename="classified_output.csv")
    finally:
        await file.close()
