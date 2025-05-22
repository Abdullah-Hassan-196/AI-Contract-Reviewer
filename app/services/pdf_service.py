import fitz
from typing import Tuple, List, Dict, Any, Optional
import logging
import shutil
import os
import pytesseract
from PIL import Image, ImageEnhance
import tempfile
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
from difflib import SequenceMatcher

# Configure logging
logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        # Configure Tesseract for better text extraction
        self.tesseract_config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = False, preserve_layout: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text from a PDF file with enhanced formatting preservation.
        
        Parameters:
            pdf_path (str): Path to the PDF file
            use_ocr (bool): Force OCR processing even if text extraction is available
            preserve_layout (bool): Attempt to preserve text layout and formatting
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = ""
        text_blocks = []
        
        # First try normal text extraction
        if not use_ocr:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                if preserve_layout:
                    # Extract text with layout information
                    text_dict = page.get_text("dict")
                    formatted_text = self._extract_formatted_text_from_dict(text_dict)
                else:
                    formatted_text = page.get_text()
                
                # Check if the page has meaningful text content
                if formatted_text.strip() and len(formatted_text.strip()) > 20:
                    full_text += formatted_text + "\n"
                    text_blocks.append({
                        "text": formatted_text,
                        "page": page_num + 1
                    })
        
        # If no meaningful text was extracted or OCR is forced, use enhanced OCR
        if not full_text.strip() or use_ocr:
            logger.info(f"PDF appears to be scanned or OCR was requested. Applying enhanced OCR...")
            full_text, text_blocks = self._apply_enhanced_ocr_to_pdf(doc, preserve_layout)
        
        doc.close()
        logger.info(f"Extracted text from {len(text_blocks)} pages")
        return full_text.strip(), text_blocks
    
    def _extract_formatted_text_from_dict(self, text_dict: dict) -> str:
        """
        Extract text from PyMuPDF text dictionary while preserving some formatting.
        """
        formatted_text = ""
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                block_text = ""
                prev_line_bottom = None
                
                for line in block["lines"]:
                    line_text = ""
                    prev_span_right = None
                    
                    for span in line["spans"]:
                        span_text = span["text"]
                        span_bbox = span["bbox"]
                        
                        # Add spacing between spans if there's a gap
                        if prev_span_right and span_bbox[0] - prev_span_right > 10:
                            line_text += "  "
                        
                        line_text += span_text
                        prev_span_right = span_bbox[2]
                    
                    # Add line breaks based on vertical spacing
                    if prev_line_bottom and line["bbox"][1] - prev_line_bottom > 5:
                        block_text += "\n"
                    
                    block_text += line_text.strip()
                    if line_text.strip():
                        block_text += "\n"
                    
                    prev_line_bottom = line["bbox"][3]
                
                formatted_text += block_text + "\n"
        
        return formatted_text

    def _apply_enhanced_ocr_to_pdf(self, doc: fitz.Document, preserve_layout: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Apply enhanced OCR to a PDF document with better formatting preservation.
        """
        full_text = ""
        text_blocks = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for page_num in range(len(doc)):
                logger.info(f"Processing page {page_num + 1} with enhanced OCR")
                page = doc[page_num]
                
                # Get the page as a high-quality image
                pix = page.get_pixmap(alpha=False, matrix=fitz.Matrix(3, 3))  # better quality ~300 DPI
                img_path = os.path.join(temp_dir, f"page_{page_num}.png")
                pix.save(img_path)
                
                try:
                    # Preprocess image for better OCR
                    processed_img_path = self._preprocess_image_for_ocr(img_path, temp_dir, page_num)
                    
                    if preserve_layout:
                        page_text = self._extract_text_with_layout(processed_img_path)
                    else:
                        image = Image.open(processed_img_path)
                        page_text = pytesseract.image_to_string(image, config=self.tesseract_config)
                    
                    full_text += page_text + "\n"
                    text_blocks.append({
                        "text": page_text,
                        "page": page_num + 1
                    })
                    
                except Exception as e:
                    logger.error(f"Enhanced OCR error on page {page_num + 1}: {e}")
        
        return full_text, text_blocks
    
    def _preprocess_image_for_ocr(self, img_path: str, temp_dir: str, page_num: int) -> str:
        """
        Preprocess image to improve OCR accuracy using CLAHE, denoising, and thresholding.
        """
        # Load image
        img = cv2.imread(img_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(contrast, h=30)

        # Thresholding with Otsu
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Save the processed image
        processed_path = os.path.join(temp_dir, f"processed_page_{page_num}.png")
        cv2.imwrite(processed_path, cleaned)

        return processed_path
    
    def _extract_text_with_layout(self, img_path: str) -> str:
        """
        Extract text while attempting to preserve layout using Tesseract's hOCR output.
        """
        try:
            # Get detailed OCR data with bounding boxes
            image = Image.open(img_path)
            
            # Use pytesseract to get detailed data
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config=self.tesseract_config
            )
            
            # Reconstruct text with layout information
            formatted_text = self._reconstruct_layout_from_ocr_data(data)
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"Error in layout-aware OCR: {e}")
            # Fallback to basic OCR
            image = Image.open(img_path)
            return pytesseract.image_to_string(image, config=self.tesseract_config)
    
    def _reconstruct_layout_from_ocr_data(self, data: dict) -> str:
        """
        Reconstruct text layout from Tesseract OCR data.
        """
        # Group words by lines based on their vertical position
        lines = {}
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Only consider words with decent confidence
                word = data['text'][i].strip()
                if word:
                    top = data['top'][i]
                    left = data['left'][i]
                    
                    # Group words into lines (allowing some vertical tolerance)
                    line_key = None
                    for existing_top in lines.keys():
                        if abs(top - existing_top) <= 10:  # Tolerance for line grouping
                            line_key = existing_top
                            break
                    
                    if line_key is None:
                        line_key = top
                        lines[line_key] = []
                    
                    lines[line_key].append((left, word))
        
        # Sort lines by vertical position and words by horizontal position
        formatted_text = ""
        for line_top in sorted(lines.keys()):
            words_in_line = sorted(lines[line_top], key=lambda x: x[0])
            
            # Reconstruct line with appropriate spacing
            line_text = ""
            prev_right = None
            
            for left, word in words_in_line:
                if prev_right and left - prev_right > 50:  # Large gap indicates column separation
                    line_text += "    "  # Add tab-like spacing
                elif prev_right and left - prev_right > 20:  # Medium gap
                    line_text += "  "   # Add double space
                elif line_text:  # Normal spacing
                    line_text += " "
                
                line_text += word
                # Estimate word width (rough approximation)
                prev_right = left + len(word) * 8
            
            formatted_text += line_text + "\n"
        
        return formatted_text
    
    def convert_scanned_pdf_with_layout(self, input_pdf_path: str, output_pdf_path: str, dpi: int = 300) -> bool:
        """
        Convert a scanned PDF to a searchable PDF while preserving layout as much as possible.
        Creates a hybrid PDF with the original image as background and invisible searchable text overlay.
        """
        try:
            logger.info(f"Converting scanned PDF to searchable PDF with layout preservation: {input_pdf_path}")
            
            if not os.path.exists(input_pdf_path):
                logger.error(f"Input PDF file does not exist: {input_pdf_path}")
                return False
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_pdf_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Open original PDF to preserve page dimensions
            original_doc = fitz.open(input_pdf_path)
            result_doc = fitz.open()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for page_num in range(len(original_doc)):
                    logger.info(f"Processing page {page_num+1}/{len(original_doc)} with layout-aware OCR")
                    
                    try:
                        original_page = original_doc[page_num]
                        page_rect = original_page.rect
                        
                        # Get high-quality image of the page
                        pix = original_page.get_pixmap(alpha=False, matrix=fitz.Matrix(2, 2))
                        img_path = os.path.join(temp_dir, f"page_{page_num}.png")
                        pix.save(img_path)
                        
                        # Preprocess image for better OCR
                        processed_img_path = self._preprocess_image_for_ocr(img_path, temp_dir, page_num)
                        
                        # Get OCR data with bounding boxes
                        image = Image.open(processed_img_path)
                        ocr_data = pytesseract.image_to_data(
                            image, 
                            output_type=pytesseract.Output.DICT,
                            config=self.tesseract_config
                        )
                        
                        # Create new page with original dimensions
                        new_page = result_doc.new_page(width=page_rect.width, height=page_rect.height)
                        
                        # Insert original image as background
                        new_page.insert_image(page_rect, pixmap=pix)
                        
                        # Add invisible text layer based on OCR coordinates
                        self._add_invisible_text_layer(new_page, ocr_data, page_rect, (pix.width, pix.height))
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page_num+1}: {e}")
                        # Add original page without OCR if processing fails
                        result_doc.insert_pdf(original_doc, from_page=page_num, to_page=page_num)
                        continue
                
                if result_doc.page_count > 0:
                    result_doc.save(output_pdf_path, garbage=4, deflate=True)
                    logger.info(f"Successfully converted PDF with layout preservation: {output_pdf_path}")
                    return True
                else:
                    logger.error("No pages were successfully processed")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in layout-preserving PDF conversion: {str(e)}")
            return False
        finally:
            if 'original_doc' in locals():
                original_doc.close()
            if 'result_doc' in locals():
                result_doc.close()
    
    def check_if_scanned(self, pdf_path: str) -> bool:
        """
        Check if a PDF is likely scanned (image-based) by testing for lack of extractable text.
        """
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text().strip()
                if len(text) > 30:
                    doc.close()
                    return False  # Text found, not scanned
            doc.close()
            return True  # No real text found, likely scanned
        except Exception as e:
            logger.error(f"Error checking if PDF is scanned: {e}")
            return True  # Assume scanned if there's an error
        
    def convert_scanned_pdf(self, input_pdf_path: str, output_pdf_path: str, dpi: int = 300):
        return self.convert_scanned_pdf_with_layout(input_pdf_path, output_pdf_path, dpi)
    
    def _add_invisible_text_layer(self, page: fitz.Page, ocr_data: dict, page_rect: fitz.Rect, img_size: tuple):
        """
        Add invisible text layer to page based on OCR data with proper coordinate mapping.
        """
        img_width, img_height = img_size
        scale_x = page_rect.width / img_width
        scale_y = page_rect.height / img_height
        
        # Group words by lines for better text positioning
        lines = {}
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:  # Only consider words with decent confidence
                word = ocr_data['text'][i].strip()
                if word:
                    top = ocr_data['top'][i]
                    left = ocr_data['left'][i]
                    width = ocr_data['width'][i]
                    height = ocr_data['height'][i]
                    
                    # Group words into lines
                    line_key = None
                    for existing_top in lines.keys():
                        if abs(top - existing_top) <= 10:
                            line_key = existing_top
                            break
                    
                    if line_key is None:
                        line_key = top
                        lines[line_key] = []
                    
                    lines[line_key].append({
                        'word': word,
                        'left': left,
                        'top': top,
                        'width': width,
                        'height': height
                    })
        
        # Add text for each line
        for line_top in sorted(lines.keys()):
            words_in_line = sorted(lines[line_top], key=lambda x: x['left'])
            
            # Combine words in line
            line_text = " ".join([w['word'] for w in words_in_line])
            
            if line_text.strip():
                # Use first word's position for line positioning
                first_word = words_in_line[0]
                
                # Convert OCR coordinates to PDF coordinates
                x = first_word['left'] * scale_x
                y = page_rect.height - (first_word['top'] + first_word['height']) * scale_y
                
                # Estimate appropriate font size based on OCR height
                font_size = max(8, first_word['height'] * scale_y * 0.8)
                
                # Insert invisible text
                text_rect = fitz.Rect(x, y, x + len(line_text) * font_size * 0.6, y + font_size)
                page.insert_text(
                    (x, y + font_size),  # Bottom-left position for text
                    line_text,
                    fontsize=font_size,
                    color=(1, 1, 1),  # White text (invisible on white background)
                    render_mode=3  # Neither fill nor stroke (invisible)
                )
    
    def _normalize_text_for_search(self, text: str) -> str:
        """
        Normalize text for better matching by removing extra whitespace and normalizing punctuation.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        return text
    
    def _find_similar_text(self, page_text: str, search_text: str, threshold: float = 0.6) -> List[str]:
        """
        Find similar text in page using fuzzy matching.
        """
        search_text_norm = self._normalize_text_for_search(search_text)
        
        # Split page text into sentences and paragraphs
        sentences = re.split(r'[.!?]\s+', page_text)
        
        matches = []
        for sentence in sentences:
            sentence_norm = self._normalize_text_for_search(sentence)
            
            # Check similarity
            similarity = SequenceMatcher(None, search_text_norm.lower(), sentence_norm.lower()).ratio()
            if similarity >= threshold:
                matches.append(sentence.strip())
            
            # Also check if search text is a substring of the sentence
            if search_text_norm.lower() in sentence_norm.lower():
                matches.append(sentence.strip())
        
        return matches
    
    def highlight_contradictions(self, pdf_path: str, highlights: List[Dict[str, Any]], output_path: str):
        """
        Enhanced highlight contradictions in a PDF file with better text matching for OCR PDFs.
        """
        if not highlights:
            logger.warning(f"No highlights provided for {pdf_path}")
            shutil.copy(pdf_path, output_path)
            return

        logger.info(f"Highlighting {len(highlights)} contradictions in {pdf_path}")
        doc = fitz.open(pdf_path)
        highlighted_count = 0

        for highlight in highlights:
            text = highlight.get("text", "").strip()
            if not text or text == "MISSING":
                continue

            found_match = False
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # First try exact search
                matches = page.search_for(text, quads=True)
                
                if matches:
                    for rect in matches:
                        highlight_annot = page.add_highlight_annot(rect)
                        highlight_annot.set_colors(stroke=(1, 0.85, 0))
                        highlight_annot.update()
                        found_match = True
                        highlighted_count += 1
                
                # If no exact match, try normalized search
                if not matches:
                    normalized_text = self._normalize_text_for_search(text)
                    matches = page.search_for(normalized_text, quads=True)
                    
                    if matches:
                        for rect in matches:
                            highlight_annot = page.add_highlight_annot(rect)
                            highlight_annot.set_colors(stroke=(1, 0.85, 0))
                            highlight_annot.update()
                            found_match = True
                            highlighted_count += 1
                
                # If still no match, try partial matching for longer texts
                if not matches and len(text) > 20:
                    words = text.split()
                    if len(words) > 3:
                        # Try different combinations of words
                        for start_idx in range(min(3, len(words))):
                            for end_idx in range(max(3, start_idx + 1), min(len(words) + 1, start_idx + 6)):
                                partial_text = " ".join(words[start_idx:end_idx])
                                if len(partial_text) > 10:
                                    matches = page.search_for(partial_text, quads=True)
                                    if matches:
                                        for rect in matches:
                                            highlight_annot = page.add_highlight_annot(rect)
                                            highlight_annot.set_colors(stroke=(1, 0.85, 0))
                                            highlight_annot.update()
                                            found_match = True
                                            highlighted_count += 1
                                        break
                            if matches:
                                break
                
                # For OCR PDFs, try fuzzy matching as last resort
                if not found_match:
                    page_text = page.get_text()
                    if page_text:
                        similar_matches = self._find_similar_text(page_text, text, threshold=0.7)
                        for similar_text in similar_matches:
                            matches = page.search_for(similar_text, quads=True)
                            if matches:
                                for rect in matches:
                                    highlight_annot = page.add_highlight_annot(rect)
                                    highlight_annot.set_colors(stroke=(1, 0.85, 0))
                                    highlight_annot.update()
                                    found_match = True
                                    highlighted_count += 1
                                break
                        
                if found_match:
                    break
            
            if not found_match:
                logger.warning(f"Could not find text to highlight: {text[:50]}...")

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        logger.info(f"Saved highlighted PDF to {output_path} with {highlighted_count} highlights applied")