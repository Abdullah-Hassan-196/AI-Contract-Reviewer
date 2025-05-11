import fitz
from typing import Tuple, List
import numpy as np

class PDFService:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[dict]]:
        """
        Extract text and its coordinates from a PDF file.
        Returns the full text and a list of text blocks with their coordinates.
        """
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
                                full_text += text + "\n"
                                text_blocks.append({
                                    "text": text,
                                    "page": page_num + 1,
                                    "bbox": span["bbox"],
                                    "font": span["font"],
                                    "size": span["size"]
                                })
        
        doc.close()
        return full_text.strip(), text_blocks

    @staticmethod
    def highlight_contradictions(pdf_path: str, highlights: List[dict], output_path: str):
        """
        Highlight contradictions in a PDF file.
        """
        doc = fitz.open(pdf_path)
        
        for highlight in highlights:
            page_num = highlight["page"] - 1  # Convert to 0-based index
            if page_num < len(doc):
                page = doc[page_num]
                bbox = highlight["bbox"]
                
                # Create a highlight annotation
                annot = page.add_highlight_annot(bbox)
                if annot:
                    # Set highlight properties
                    annot.set_colors(stroke=(1, 0, 0))  # Red color
                    annot.set_opacity(0.3)  # Semi-transparent
                    annot.update()
        
        # Save the modified PDF
        doc.save(output_path)
        doc.close() 