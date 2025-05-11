from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/compare-documents")
async def compare_documents(
    main_document: UploadFile = File(...),
    target_document: UploadFile = File(...)
):
    # Initialize variables
    main_path = None
    target_path = None
    main_output = None
    target_output = None
    
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
        analysis_result = ai_service.analyze_documents(main_text, target_text, main_blocks, target_blocks)

        # Highlight contradictions in both documents
        main_output = TEMP_DIR / "main_highlighted.pdf"
        target_output = TEMP_DIR / "target_highlighted.pdf"

        # Create highlighted versions
        pdf_service.highlight_contradictions(str(main_path), analysis_result["main_highlights"], str(main_output))
        pdf_service.highlight_contradictions(str(target_path), analysis_result["target_highlights"], str(target_output))

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
        for file_path in [main_path, target_path, main_output, target_output]:
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error deleting file {file_path}: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=filename) 