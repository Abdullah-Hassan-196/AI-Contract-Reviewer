import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
import logging
import os
from typing import Optional, Tuple
import shutil

logger = logging.getLogger(__name__)

class PDFConverter:
    def __init__(self, tesseract_path: Optional[str] = None, poppler_path: Optional[str] = None):
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
        self.poppler_path = poppler_path or (
            r"C:\Program Files\poppler-24.08.0\Library\bin"
        )
        logger.info("PDFConverter initialized with Tesseract OCR")

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

    def process_pdf(self, input_path: str, output_dir: str = "temp") -> Tuple[bool, Optional[str]]:
        """
        Process a PDF file - convert to searchable if scanned, otherwise return as is.
        
        Args:
            input_path: Path to input PDF file
            output_dir: Directory to save output PDF
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            if self.is_scanned_pdf(input_path):
                logger.info(f"PDF appears to be scanned, converting to searchable: {input_path}")
                return self.convert_to_text_pdf(input_path, output_dir)
            else:
                logger.info(f"PDF is already text-based, copying to output: {input_path}")
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Copy the file to output directory
                input_filename = os.path.basename(input_path)
                output_filename = f"text_{input_filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                shutil.copy2(input_path, output_path)
                return True, output_path
                
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            logger.error(error_msg)
            return False, None

    def convert_to_text_pdf(
        self, 
        input_path: str,
        output_dir: str = "temp"
    ) -> Tuple[bool, Optional[str]]:
        """
        Convert a scanned PDF to a searchable PDF.
        
        Args:
            input_path: Path to input PDF file
            output_dir: Directory to save output PDF
            
        Returns:
            Tuple of (success, output_path)
        """
        try:
            logger.info(f"Converting PDF to searchable format: {input_path}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output path
            input_filename = os.path.basename(input_path)
            output_filename = f"searchable_{input_filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Convert PDF to images
            images = self._convert_pdf_to_images(input_path)
            
            # Convert images to searchable PDF
            self._images_to_searchable_pdf(images, output_path)
            
            logger.info(
                f"Successfully converted PDF to searchable format: {output_path}"
            )
            return True, output_path
            
        except Exception as e:
            error_msg = f"Error converting PDF: {str(e)}"
            logger.error(error_msg)
            return False, None

    def _convert_pdf_to_images(self, pdf_path: str, dpi: int = 300):
        """Convert scanned PDF to images (one per page)."""
        return convert_from_path(
            pdf_path,
            dpi=dpi,
            poppler_path=self.poppler_path
        )

    def _images_to_searchable_pdf(self, images, output_pdf_path: str):
        """Perform OCR and merge images into a searchable PDF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            combined_pdf = None

            for idx, image in enumerate(images):
                # Convert image to PDF with OCR
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                    image,
                    extension='pdf'
                )
                temp_pdf_path = temp_dir_path / f"page_{idx}.pdf"
                
                with open(temp_pdf_path, 'wb') as f:
                    f.write(pdf_bytes)

                # Combine PDFs
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