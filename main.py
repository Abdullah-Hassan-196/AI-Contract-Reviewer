from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from typing import List
import tempfile
import shutil
from pathlib import Path

from app.services.pdf_service import PDFService
from app.services.ai_service import AIService

app = FastAPI(
    title="AI Contract Reviewer",
    description="API for AI-powered contract review system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_service = PDFService()
ai_service = AIService()

# Create temporary directory for file processing
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to AI Contract Reviewer API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/compare-documents")
async def compare_documents(
    main_document: UploadFile = File(...),
    target_document: UploadFile = File(...)
):
    try:
        # Save uploaded files temporarily
        main_path = TEMP_DIR / "main.pdf"
        target_path = TEMP_DIR / "target.pdf"
        
        with open(main_path, "wb") as buffer:
            shutil.copyfileobj(main_document.file, buffer)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(target_document.file, buffer)

        # Extract text from both documents
        main_text, main_blocks = pdf_service.extract_text_from_pdf(str(main_path))
        target_text, target_blocks = pdf_service.extract_text_from_pdf(str(target_path))

        # Analyze documents using AI
        analysis_result = ai_service.analyze_documents(main_text, target_text)

        # Highlight contradictions in both documents
        main_output = TEMP_DIR / "main_highlighted.pdf"
        target_output = TEMP_DIR / "target_highlighted.pdf"

        # Find blocks that contain contradictory segments
        main_contradictions = []
        target_contradictions = []

        for segment in analysis_result["contradictory_segments"]:
            for block in main_blocks:
                if segment.lower() in block["text"].lower():
                    main_contradictions.append(block)
            for block in target_blocks:
                if segment.lower() in block["text"].lower():
                    target_contradictions.append(block)

        # Create highlighted versions
        pdf_service.highlight_contradictions(str(main_path), main_contradictions, str(main_output))
        pdf_service.highlight_contradictions(str(target_path), target_contradictions, str(target_output))

        return {
            "analysis": analysis_result,
            "highlighted_documents": {
                "main": str(main_output),
                "target": str(target_output)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary files
        if main_path.exists():
            main_path.unlink()
        if target_path.exists():
            target_path.unlink()
        if main_output.exists():
            main_output.unlink()
        if target_output.exists():
            target_output.unlink()

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=filename) 