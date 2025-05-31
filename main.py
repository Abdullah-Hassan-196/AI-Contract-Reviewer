from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
from pathlib import Path
import os
import asyncio
import aiofiles
from datetime import datetime, timedelta

from app.services.pdf_service import PDFService
from app.services.ai_service import AIService
from app.services.pdf_converter import PDFConverter

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
pdf_converter = PDFConverter(
    poppler_path=r"C:\Program Files\poppler-24.08.0\Library\bin"  # Update this path to match your Poppler installation
)

# Create temporary directory for file processing
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

async def cleanup_file(file_path: Path):
    """Asynchronously clean up a file"""
    try:
        if file_path and file_path.exists():
            await asyncio.to_thread(file_path.unlink)
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")

async def save_upload_file(upload_file: UploadFile, filename: str) -> Path:
    """Asynchronously save an uploaded file"""
    file_path = TEMP_DIR / filename
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    return file_path

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/compare")
async def compare_documents(
    background_tasks: BackgroundTasks,
    main_document: UploadFile = File(...),
    target_document: UploadFile = File(...)
):
    """
    Compare two PDF documents and highlight contradictions.
    First converts scanned PDFs to text-based PDFs if needed.
    
    Returns an analysis of the documents along with links to highlighted versions.
    """
    main_path = None
    target_path = None
    main_output = None
    target_output = None
    
    try:
        # Generate unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_filename = (
            f"main_{timestamp}_{os.urandom(4).hex()}_{main_document.filename}"
        )
        target_filename = (
            f"target_{timestamp}_{os.urandom(4).hex()}_{target_document.filename}"
        )
        
        # Save uploaded files asynchronously
        main_path, target_path = await asyncio.gather(
            save_upload_file(main_document, main_filename),
            save_upload_file(target_document, target_filename)
        )
        
        logger.info(f"Files saved: {main_path}, {target_path}")
        
        # Process PDFs in parallel
        pdf_converter = PDFConverter()
        
        # Process both documents concurrently
        try:
            logger.info(f"Processing main document: {main_path}")
            main_result = await pdf_converter.process_pdf(str(main_path))
            logger.info(f"Main document processing result: {main_result}")
            
            logger.info(f"Processing target document: {target_path}")
            target_result = await pdf_converter.process_pdf(str(target_path))
            logger.info(f"Target document processing result: {target_result}")
            
            if not main_result[0] or not target_result[0]:
                error_msg = (
                    f"PDF processing failed. Main success: {main_result[0]}, "
                    f"Target success: {target_result[0]}"
                )
                logger.error(error_msg)
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
                
            processed_main_path, processed_target_path = main_result[1], target_result[1]
            logger.info(
                f"Successfully processed PDFs. Main: {processed_main_path}, "
                f"Target: {processed_target_path}"
            )
        except Exception as e:
            logger.error(f"Error processing PDFs: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDFs: {str(e)}"
            )
        
        # Extract text from processed documents concurrently
        main_text_task = asyncio.create_task(
            asyncio.to_thread(pdf_service.extract_text_from_pdf, processed_main_path)
        )
        target_text_task = asyncio.create_task(
            asyncio.to_thread(pdf_service.extract_text_from_pdf, processed_target_path)
        )
        
        main_result, target_result = await asyncio.gather(main_text_task, target_text_task)
        main_text, main_blocks = main_result
        target_text, target_blocks = target_result
        
        # Analyze documents using AI
        analysis_result = await ai_service.analyze_documents(
            main_text,
            target_text,
            main_blocks,
            target_blocks
        )
        
        # Generate output filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_filename = (
            f"main_highlighted_{timestamp}_{os.urandom(4).hex()}.pdf"
        )
        target_output_filename = (
            f"target_highlighted_{timestamp}_{os.urandom(4).hex()}.pdf"
        )
        
        # Set output paths
        main_output = TEMP_DIR / main_output_filename
        target_output = TEMP_DIR / target_output_filename
        
        # Create highlighted versions concurrently
        highlight_tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    pdf_service.highlight_contradictions,
                    processed_main_path,
                    analysis_result["main_highlights"],
                    str(main_output)
                )
            ),
            asyncio.create_task(
                asyncio.to_thread(
                    pdf_service.highlight_contradictions,
                    processed_target_path,
                    analysis_result["target_highlights"],
                    str(target_output)
                )
            )
        ]
        await asyncio.gather(*highlight_tasks)
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_file, main_path)
        background_tasks.add_task(cleanup_file, target_path)
        background_tasks.add_task(cleanup_file, Path(processed_main_path))
        background_tasks.add_task(cleanup_file, Path(processed_target_path))
        
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
        
        logger.info(
            f"Analysis complete with {len(analysis_result.get('contradictions', []))} "
            "contradictions"
        )
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error in compare-documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/temp/{file_path:path}")
async def download_file(file_path: str):
    """
    Download a processed file from the temp directory.
    """
    full_path = TEMP_DIR / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=str(full_path),
        filename=os.path.basename(file_path),
        media_type="application/pdf"
    )

@app.post("/convert-pdf")
async def convert_pdf(
    background_tasks: BackgroundTasks,
    document: UploadFile = File(...)
):
    """
    Convert a scanned PDF to a text-based PDF while preserving the original 
    formatting.
    
    Returns the path to the converted PDF.
    """
    input_path = None
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{os.urandom(4).hex()}_{document.filename}"
        input_path = TEMP_DIR / filename
        
        # Save uploaded file asynchronously
        async with aiofiles.open(input_path, 'wb') as out_file:
            content = await document.read()
            await out_file.write(content)
        
        logger.info(f"File saved: {input_path}")
        
        # Convert PDF
        pdf_converter = PDFConverter()
        success, output_path = await asyncio.to_thread(
            pdf_converter.convert_to_text_pdf,
            str(input_path)
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to convert PDF"
            )
        
        # Schedule cleanup of input file
        background_tasks.add_task(cleanup_file, input_path)
        
        return {"output_path": output_path}
        
    except Exception as e:
        logger.error(f"Error converting PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.on_event("startup")
async def startup_event():
    """
    Run when the server starts.
    """
    logger.info("Starting AI Contract Reviewer API")
    
    # Ensure temp directory exists
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Start background task for cleaning up old files
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """
    Periodically clean up old files in the temp directory.
    """
    while True:
        try:
            await cleanup_old_files()
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {str(e)}")
        await asyncio.sleep(3600)  # Run every hour

async def cleanup_old_files(max_age_hours: int = 24):
    """
    Clean up files older than max_age_hours in the temp directory.
    """
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for file_path in TEMP_DIR.glob("*"):
        try:
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age < cutoff_time:
                    await cleanup_file(file_path)
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)