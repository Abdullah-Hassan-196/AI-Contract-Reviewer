import fitz
from typing import Tuple, List, Dict, Any
import logging
import shutil

# Configure logging
logger = logging.getLogger(__name__)

class PDFService:
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
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text += text + "\n"
            text_blocks.append({
                "text": text,
                "page": page_num + 1
            })
        
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
