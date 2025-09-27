# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import base64
import io
import os

app = FastAPI(title="MoSPI AI Survey API")


origins = [
    "https://webcade.vercel.app/",  # <-- Replace with actual Vercel URL
                     # for local testing (optional)
]
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_STORE = {}

class Base64File(BaseModel):
    filename: str
    data: str

# ---------- Utility Functions ----------
def to_python_dict(d):
    """Convert all numpy values to native Python types"""
    return {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in d.items()}

def read_file(file: UploadFile = None, base64_file: Base64File = None):
    if file:
        filename = file.filename
        content = file.file.read()
    elif base64_file:
        filename = base64_file.filename
        try:
            content = base64.b64decode(base64_file.data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 data")
    else:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    return df

def validate_data(df: pd.DataFrame):
    return {
        "missing_values": to_python_dict(df.isnull().sum().to_dict()),
        "duplicates": int(df.duplicated().sum())
    }

def traditional_process(df: pd.DataFrame):
    df_cleaned = df.copy()
    df_cleaned.fillna(df_cleaned.mean(numeric_only=True), inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    return df_cleaned

def ai_hybrid_process(df: pd.DataFrame, numerical_method="Smart Mean/Median", text_method="AI Text Generation"):
    df_processed = df.copy()
    df_processed.fillna(df_processed.mean(numeric_only=True), inplace=True)
    for col in df_processed.select_dtypes(include="object").columns:
        df_processed[col].fillna("AI_generated_value", inplace=True)
    ai_report = {
        "numerical_method": numerical_method,
        "text_method": text_method,
        "missing_filled": int(df.isnull().sum().sum())
    }
    return df_processed, ai_report

def generate_report(df: pd.DataFrame, ai_report: dict):
    report = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "ai_summary": ai_report,
        "missing_values_after_processing": to_python_dict(df.isnull().sum().to_dict())
    }
    return report

# ---------- API Endpoints ----------
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(None), base64_file: Base64File = None):
    try:
        df = read_file(file, base64_file)
        validation_results = validate_data(df)
        DATA_STORE["raw_data"] = df
        return {"message": "File uploaded successfully", "validation": validation_results}
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/traditional-process")
async def traditional_processing():
    df = DATA_STORE.get("raw_data")
    if df is None:
        return JSONResponse(status_code=400, content={"error": "No data uploaded"})
    df_cleaned = traditional_process(df)
    DATA_STORE["traditional_data"] = df_cleaned
    return {"message": "Traditional processing complete",
            "rows": int(df_cleaned.shape[0]),
            "columns": int(df_cleaned.shape[1])}

@app.post("/ai-process")
async def ai_process(numerical_method: str = Form("Smart Mean/Median"), text_method: str = Form("AI Text Generation")):
    df = DATA_STORE.get("raw_data")
    if df is None:
        return JSONResponse(status_code=400, content={"error": "No data uploaded"})
    df_processed, ai_report = ai_hybrid_process(df, numerical_method, text_method)
    DATA_STORE["ai_processed_data"] = df_processed
    DATA_STORE["ai_report"] = ai_report
    return {"message": "AI processing complete",
            "rows": int(df_processed.shape[0]),
            "columns": int(df_processed.shape[1]),
            "ai_report": ai_report}

@app.get("/missing-analysis")
async def missing_analysis():
    df = DATA_STORE.get("raw_data")
    if df is None:
        return JSONResponse(status_code=400, content={"error": "No data uploaded"})
    return {"missing_values": to_python_dict(df.isnull().sum().to_dict())}

@app.post("/estimate")
async def estimate():
    df = DATA_STORE.get("ai_processed_data")
    if df is None:
        return JSONResponse(status_code=400, content={"error": "AI processing not done yet"})
    estimation = to_python_dict(df.select_dtypes(include=np.number).sum().to_dict())
    return {"estimation": estimation}

@app.post("/generate-report")
async def generate_report_endpoint():
    df = DATA_STORE.get("ai_processed_data")
    ai_report = DATA_STORE.get("ai_report")
    if df is None or ai_report is None:
        return JSONResponse(status_code=400, content={"error": "AI processing not done yet"})
    report = generate_report(df, ai_report)
    DATA_STORE["report"] = report
    return report

@app.get("/download-processed")
async def download_processed():
    df = DATA_STORE.get("ai_processed_data")
    if df is None:
        return JSONResponse(status_code=404, content={"error": "No processed data available. Please run AI processing first."})
    
    file_path = "processed_data.csv"
    df.to_csv(file_path, index=False)
    return FileResponse(file_path, filename="ai_processed_data.csv", media_type="text/csv")

