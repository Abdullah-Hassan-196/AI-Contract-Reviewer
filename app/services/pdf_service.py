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
        Extract text and its coordinates from a PDF file.
        Returns the full text and a list of text blocks with their coordinates.
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = ""
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            if text.strip():
                                full_text += text + " "  # Using space instead of newline for better context
                                text_blocks.append({
                                    "text": text,
                                    "page": page_num + 1,
                                    "bbox": span["bbox"],
                                    "font": span["font"],
                                    "size": span["size"]
                                })
        
        doc.close()
        logger.info(f"Extracted {len(text_blocks)} text blocks from PDF")
        return full_text.strip(), text_blocks
    
    @staticmethod
    def find_text_in_blocks(search_text: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find all occurrences of text in the text blocks and return their locations.
        """
        logger.info(f"Searching for text: '{search_text[:50]}...' in {len(blocks)} blocks")
        matching_blocks = []
        
        # Normalize search text for comparison
        search_text = search_text.strip().lower()
        
        # Try exact match first
        for block in blocks:
            block_text = block["text"].strip().lower()
            if search_text == block_text:
                matching_blocks.append(block)
        
        # If no exact match, try substring match
        if not matching_blocks:
            for block in blocks:
                block_text = block["text"].strip().lower()
                if search_text in block_text or block_text in search_text:
                    matching_blocks.append(block)
        
        logger.info(f"Found {len(matching_blocks)} matching blocks")
        return matching_blocks
    
    @staticmethod
    def highlight_contradictions(pdf_path: str, highlights: List[Dict[str, Any]], output_path: str):
        """
        Highlight contradictions in a PDF file using built-in highlighting with default yellow color.
        Searches the entire document for the given text and applies highlight.
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
                logger.warning("Empty text provided in highlight entry, skipping.")
                continue

            for page_num in range(len(doc)):
                page = doc[page_num]
                matches = page.search_for(text)  # Search the page for the text
                if matches:
                    for rect in matches:
                        annot = page.add_highlight_annot(rect)
                        if annot:
                            annot.update()
                            logger.info(f"Highlighted '{text}' on page {page_num + 1} at {rect}")
                        else:
                            logger.warning(f"Failed to add highlight for '{text}' on page {page_num + 1}")
                else:
                    logger.warning(f"Text '{text}' not found on page {page_num + 1}")

        doc.save(output_path)
        doc.close()
        logger.info(f"Saved highlighted PDF to {output_path}")
