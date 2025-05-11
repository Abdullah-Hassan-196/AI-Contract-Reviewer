from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import shutil
from pathlib import Path
import os

from app.services.pdf_service import PDFService
from app.services.ai_service import AIService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
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
ai_service = AIService()
pdf_service = PDFService()

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
    """
    Compare two PDF documents and highlight contradictions.
    
    Returns an analysis of the documents along with links to highlighted versions.
    """
    # Initialize variables
    main_path = None
    target_path = None
    main_output = None
    target_output = None
    
    try:
        # Generate unique filenames to prevent collisions
        main_filename = f"main_{os.urandom(4).hex()}_{main_document.filename}"
        target_filename = f"target_{os.urandom(4).hex()}_{target_document.filename}"
        
        # Save uploaded files temporarily
        main_path = TEMP_DIR / main_filename
        target_path = TEMP_DIR / target_filename
        
        with open(main_path, "wb") as buffer:
            shutil.copyfileobj(main_document.file, buffer)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(target_document.file, buffer)
        
        logger.info(f"Files saved: {main_path}, {target_path}")
        
        # Extract text from both documents
        main_text, main_blocks = pdf_service.extract_text_from_pdf(str(main_path))
        target_text, target_blocks = pdf_service.extract_text_from_pdf(str(target_path))
        
        # Log extracted text samples for debugging
        logger.debug(f"Main text sample: {main_text[:200]}...")
        logger.debug(f"Target text sample: {target_text[:200]}...")
        
        # Analyze documents using AI
        analysis_result = ai_service.analyze_documents(main_text, target_text, main_blocks, target_blocks)
        
        # Generate output filenames
        main_output_filename = f"main_highlighted_{os.urandom(4).hex()}.pdf"
        target_output_filename = f"target_highlighted_{os.urandom(4).hex()}.pdf"
        
        # Set output paths
        main_output = TEMP_DIR / main_output_filename
        target_output = TEMP_DIR / target_output_filename
        
        # Create highlighted versions
        pdf_service.highlight_contradictions(str(main_path), analysis_result["main_highlights"], str(main_output))
        pdf_service.highlight_contradictions(str(target_path), analysis_result["target_highlights"], str(target_output))
        
        # Include debug info in response
        response_data = {
            "analysis": analysis_result,
            "highlighted_documents": {
                "main": str(main_output),
                "target": str(target_output)
            },
            "debug": {
                "main_blocks_count": len(main_blocks),
                "target_blocks_count": len(target_blocks),
                "contradiction_count": len(analysis_result.get("contradictions", [])),
                "main_highlights_count": len(analysis_result.get("main_highlights", [])),
                "target_highlights_count": len(analysis_result.get("target_highlights", []))
            }
        }
        
        logger.info(f"Analysis complete with {len(analysis_result.get('contradictions', []))} contradictions")
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error in compare-documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # In a production environment, you might want to implement a cleanup strategy
        # For now, we're leaving files for debugging purposes
        pass


@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """
    Download a processed file.
    """
    full_path = TEMP_DIR / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=full_path,
        filename=os.path.basename(file_path),
        media_type="application/pdf"
    )


@app.on_event("startup")
async def startup_event():
    """
    Run when the server starts.
    """
    logger.info("Starting AI Contract Reviewer API")
    
    # Ensure temp directory exists
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Optional: clean old files
    # This would be better handled with a scheduled task in production
    # cleanup_old_files()


def cleanup_old_files(max_age_hours=24):
    """
    Clean up old files in the temp directory.
    """
    import time
    current_time = time.time()
    count = 0
    
    for file_path in TEMP_DIR.glob("*"):
        if file_path.is_file():
            file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
            if file_age_hours > max_age_hours:
                try:
                    file_path.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting old file {file_path}: {str(e)}")
    
    logger.info(f"Cleaned up {count} old files from temp directory")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)