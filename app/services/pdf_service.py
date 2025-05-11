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
        Highlight contradictions in a PDF file.
        """
        if not highlights:
            logger.warning(f"No highlights provided for {pdf_path}")
            # Just copy the original file
            shutil.copy(pdf_path, output_path)
            return
            
        logger.info(f"Highlighting {len(highlights)} contradictions in {pdf_path}")
        doc = fitz.open(pdf_path)
        
        for highlight in highlights:
            try:
                page_num = highlight["page"] - 1  # Convert to 0-based index
                if page_num < 0 or page_num >= len(doc):
                    logger.warning(f"Invalid page number {highlight['page']} for document with {len(doc)} pages")
                    continue
                    
                page = doc[page_num]
                bbox = highlight["bbox"]
                
                if not all(isinstance(coord, (int, float)) for coord in bbox):
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                    
                # Create a highlight annotation
                annot = page.add_highlight_annot(bbox)
                if annot:
                    logger.info(f"Added highlight on page {highlight['page']} at {bbox}")
                else:
                    logger.warning(f"Failed to add highlight on page {highlight['page']} at {bbox}")
            except Exception as e:
                logger.error(f"Error highlighting: {str(e)}")
        
        # Save the modified PDF
        doc.save(output_path)
        doc.close()
        logger.info(f"Saved highlighted PDF to {output_path}")