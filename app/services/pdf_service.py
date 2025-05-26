import fitz
from typing import Tuple, List, Dict, Any
import logging
import shutil
import base64
import io
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        # Use the same model as AIService
        self.vision_model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

    def _is_image_page(self, page) -> bool:
        """
        Check if a page contains images by looking for image blocks.
        """
        image_blocks = page.get_images()
        return len(image_blocks) > 0

    def _extract_text_from_image(self, image_data: bytes) -> str:
        """
        Extract text from an image using Gemini Vision.
        """
        try:
            # Convert image bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Create image part for Gemini
            image_parts = [{"mime_type": "image/png", "data": img_str}]
            
            # Generate content with Gemini Vision
            response = self.vision_model.generate_content(
                ["Extract all text from this image. Return only the extracted text, no explanations.", *image_parts]
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text from a PDF file.
        Returns the full text and a list of text blocks.
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = ""
        text_blocks = []
        pdf_service = PDFService()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # First try to get text directly
            text = page.get_text()
            
            # If no text found or page contains images, try OCR
            if not text.strip() or pdf_service._is_image_page(page):
                logger.info(f"Page {page_num + 1} appears to be an image or has no text. Attempting OCR...")
                
                # Get page as image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Extract text using Gemini Vision
                ocr_text = pdf_service._extract_text_from_image(img_data)
                if ocr_text:
                    text = ocr_text
                    logger.info(f"Successfully extracted text from image on page {page_num + 1}")
                else:
                    logger.warning(f"Failed to extract text from image on page {page_num + 1}")
            
            full_text += text + "\n"
            text_blocks.append({
                "text": text,
                "page": page_num + 1
            })
            
            # Print extracted text for verification
            print(f"\nPage {page_num + 1} Text:")
            print("-" * 50)
            print(text)
            print("-" * 50)
        
        doc.close()
        logger.info(f"Extracted text from {len(text_blocks)} pages")
        return full_text.strip(), text_blocks
    
    @staticmethod
    def highlight_contradictions(pdf_path: str, highlights: List[Dict[str, Any]], output_path: str):
        """
        Highlight contradictions in a PDF file using simple text highlighting.
        """
        if not highlights:
            logger.warning(f"No highlights provided for {pdf_path}")
            shutil.copy(pdf_path, output_path)
            return

        logger.info(f"Highlighting {len(highlights)} contradictions in {pdf_path}")
        doc = fitz.open(pdf_path)

        for highlight in highlights:
            text = highlight.get("text", "").strip()
            if not text:
                continue

            for page_num in range(len(doc)):
                page = doc[page_num]
                matches = page.search_for(text)
                if matches:
                    for rect in matches:
                        page.add_highlight_annot(rect)

        doc.save(output_path)
        doc.close()
        logger.info(f"Saved highlighted PDF to {output_path}")
