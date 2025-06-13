import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
import logging
import os
from typing import Optional, Tuple, List
import shutil
import hashlib
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from PIL import Image, ImageEnhance

# Configure PIL to allow larger images
Image.MAX_IMAGE_PIXELS = 933120000  # Increased limit to handle larger PDFs

logger = logging.getLogger(__name__)


class PDFConverter:
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        poppler_path: Optional[str] = None
    ):
        """
        Initialize PDFConverter with optional paths for Tesseract and Poppler.
        
        Args:
            tesseract_path: Path to Tesseract executable
            poppler_path: Path to Poppler binaries
        """
        # Set Tesseract executable path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Default Windows path
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            
        # Set Poppler path
        self.poppler_path = poppler_path
        if not self.poppler_path:
            # Try to find Poppler in common locations
            possible_paths = [
                r"C:\Program Files\poppler-24.08.0\Library\bin",
                r"C:\Program Files\poppler\Library\bin",
                r"C:\poppler-24.08.0\Library\bin",
                r"C:\poppler\Library\bin",
                os.path.join(os.path.dirname(__file__), "poppler", "bin"),
                os.path.join(os.path.dirname(__file__), "..", "poppler", "bin")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.poppler_path = path
                    break
        
        if not self.poppler_path or not os.path.exists(self.poppler_path):
            logger.warning(
                "Poppler not found in common locations. Please install Poppler "
                "and set the path manually."
            )
        
        # Initialize cache directory
        self._cache_dir = Path("temp/pdf_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"PDFConverter initialized with Tesseract OCR and Poppler path: {self.poppler_path}")
    
    def _get_cache_path(self, pdf_path: str, operation: str) -> Path:
        """Generate cache path for a PDF operation"""
        file_hash = hashlib.md5(pdf_path.encode()).hexdigest()
        return self._cache_dir / f"{operation}_{file_hash}.pdf"
    
    @lru_cache(maxsize=100)
    def _get_pdf_hash(self, pdf_path: str) -> str:
        """Get hash of PDF file for caching"""
        try:
            with open(pdf_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating PDF hash: {str(e)}")
            return ""

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Check if a PDF is scanned by attempting to extract text.
        If no text can be extracted, it's likely a scanned PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if the PDF appears to be scanned
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            # Check first few pages for text
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text += page.get_text()
                
            doc.close()
            
            # If we can't extract meaningful text, it's likely scanned
            return len(text.strip()) < 100
            
        except Exception as e:
            logger.error(f"Error checking if PDF is scanned: {str(e)}")
            return True  # Assume it's scanned if we can't check

    async def process_pdf(
        self,
        input_path: str,
        output_dir: str = "temp",
        use_cache: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Process a PDF file - convert to searchable if scanned, otherwise return as is.
        
        Args:
            input_path: Path to input PDF file
            output_dir: Directory to save output PDF
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                return False, None

            # Check cache first
            if use_cache:
                cache_path = self._get_cache_path(input_path, "processed")
                if cache_path.exists():
                    try:
                        # Create output directory if it doesn't exist
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Copy cached file to output directory
                        input_filename = os.path.basename(input_path)
                        output_filename = f"processed_{input_filename}"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        shutil.copy2(cache_path, output_path)
                        logger.info(f"Using cached processed PDF: {cache_path}")
                        return True, output_path
                    except Exception as e:
                        logger.warning(f"Error using cache: {str(e)}")
            
            # Check if PDF is scanned
            is_scanned = await asyncio.to_thread(self.is_scanned_pdf, input_path)
            logger.info(f"PDF {input_path} is {'scanned' if is_scanned else 'text-based'}")
            
            if is_scanned:
                logger.info(
                    f"PDF appears to be scanned, converting to searchable: {input_path}"
                )
                return await self.convert_to_text_pdf(input_path, output_dir, use_cache)
            else:
                logger.info(
                    f"PDF is already text-based, copying to output: {input_path}"
                )
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Copy the file to output directory
                input_filename = os.path.basename(input_path)
                output_filename = f"text_{input_filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                shutil.copy2(input_path, output_path)
                
                # Cache the result
                if use_cache:
                    try:
                        shutil.copy2(output_path, self._get_cache_path(input_path, "processed"))
                    except Exception as e:
                        logger.warning(f"Error caching result: {str(e)}")
                
                return True, output_path
                
        except Exception as e:
            error_msg = f"Error processing PDF {input_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, None

    async def convert_to_text_pdf(
        self, 
        input_path: str,
        output_dir: str = "temp",
        use_cache: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Convert a scanned PDF to a searchable PDF.
        
        Args:
            input_path: Path to input PDF file
            output_dir: Directory to save output PDF
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            # Check cache first
            if use_cache:
                cache_path = self._get_cache_path(input_path, "searchable")
                if cache_path.exists():
                    try:
                        # Create output directory if it doesn't exist
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Copy cached file to output directory
                        input_filename = os.path.basename(input_path)
                        output_filename = f"searchable_{input_filename}"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        shutil.copy2(cache_path, output_path)
                        logger.info(f"Using cached searchable PDF: {cache_path}")
                        return True, output_path
                    except Exception as e:
                        logger.warning(f"Error using cache: {str(e)}")
            
            logger.info(f"Converting PDF to searchable format: {input_path}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output path
            input_filename = os.path.basename(input_path)
            output_filename = f"searchable_{input_filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Convert PDF to images
            images = await asyncio.to_thread(
                self._convert_pdf_to_images,
                input_path
            )
            
            # Convert images to searchable PDF
            await asyncio.to_thread(
                self._images_to_searchable_pdf,
                images,
                output_path
            )
            
            # Cache the result
            if use_cache:
                try:
                    shutil.copy2(output_path, self._get_cache_path(input_path, "searchable"))
                except Exception as e:
                    logger.warning(f"Error caching result: {str(e)}")
            
            logger.info(
                f"Successfully converted PDF to searchable format: {output_path}"
            )
            return True, output_path
            
        except Exception as e:
            error_msg = f"Error converting PDF: {str(e)}"
            logger.error(error_msg)
            return False, None

    def _convert_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List:
        """
        Convert scanned PDF to images (one per page).
        
        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image conversion (default: 300 for high quality)
            
        Returns:
            List of PIL Image objects
        """
        if not self.poppler_path or not os.path.exists(self.poppler_path):
            raise ValueError(
                "Poppler is not installed or path is not set correctly. "
                "Please install Poppler and set the path in PDFConverter initialization."
            )
            
        try:
            # Convert PDF to images with high DPI
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=self.poppler_path
            )
            
            # Process each image to ensure it's not too large while maintaining quality
            processed_images = []
            max_dimension = 6000  # Increased maximum dimension
            
            for img in images:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Enhance image quality
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.5)  # Increase sharpness
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)  # Increase contrast
                
                # Check if image needs resizing
                if img.width > max_dimension or img.height > max_dimension:
                    # Calculate new dimensions maintaining aspect ratio
                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    
                    # Resize image with high-quality settings
                    img = img.resize(
                        (new_width, new_height),
                        Image.Resampling.LANCZOS
                    )
                
                processed_images.append(img)
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}", exc_info=True)
            raise ValueError(
                f"Failed to convert PDF to images. Please ensure Poppler is "
                f"installed correctly at {self.poppler_path}. Error: {str(e)}"
            )

    def _images_to_searchable_pdf(self, images: List, output_pdf_path: str) -> None:
        """
        Perform OCR and merge images into a searchable PDF.
        
        Args:
            images: List of PIL Image objects
            output_pdf_path: Path to save the output PDF
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            combined_pdf = None

            # Process images in parallel
            def process_image(args):
                idx, image = args
                # Enhance image before OCR
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.5)
                
                # Convert image to PDF with OCR
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                    image,
                    extension='pdf',
                    config='--psm 6 --oem 3'  # Use LSTM OCR Engine Mode with page segmentation mode 6
                )
                temp_pdf_path = temp_dir_path / f"page_{idx}.pdf"
                
                with open(temp_pdf_path, 'wb') as f:
                    f.write(pdf_bytes)
                return temp_pdf_path

            # Process images in parallel
            with ThreadPoolExecutor() as executor:
                temp_pdf_paths = list(executor.map(
                    process_image,
                    enumerate(images)
                ))

            # Combine PDFs
            for temp_pdf_path in temp_pdf_paths:
                if combined_pdf is None:
                    combined_pdf = fitz.open(str(temp_pdf_path))
                else:
                    temp_doc = fitz.open(str(temp_pdf_path))
                    combined_pdf.insert_pdf(temp_doc)
                    temp_doc.close()

            if combined_pdf:
                combined_pdf.save(output_pdf_path)
                combined_pdf.close()

        logger.info(f"[✓] OCR complete: {output_pdf_path}")
    
    def cleanup_cache(self, max_age_hours: int = 24) -> None:
        """
        Clean up old cache files.
        
        Args:
            max_age_hours: Maximum age of cache files in hours
        """
        try:
            for cache_file in self._cache_dir.glob("*"):
                if cache_file.is_file():
                    file_age = os.path.getmtime(cache_file)
                    if (time.time() - file_age) > (max_age_hours * 3600):
                        cache_file.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}") 