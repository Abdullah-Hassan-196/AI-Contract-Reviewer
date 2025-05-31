import fitz
from typing import Tuple, List, Dict, Any
import logging
import shutil
from functools import lru_cache
import hashlib
from pathlib import Path
import os
import time

# Configure logging
logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        """Initialize PDF service with cache settings"""
        self._cache_dir = Path("temp/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def extract_text_from_pdf(
        self, 
        pdf_path: str,
        use_cache: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text from a PDF file.
        Returns the full text and a list of text blocks.
        
        Args:
            pdf_path: Path to the PDF file
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple of (full_text, text_blocks)
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(pdf_path, "text")
            if cache_path.exists():
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = f.read().split('\n---\n')
                        if len(cached_data) == 2:
                            return cached_data[0], eval(cached_data[1])
                except Exception as e:
                    logger.warning(f"Error reading cache: {str(e)}")
        
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text.append(text)
                text_blocks.append({
                    "text": text,
                    "page": page_num + 1
                })
            
            doc.close()
            
            # Cache results
            if use_cache:
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        f.write(
                            '\n'.join(full_text) + '\n---\n' + str(text_blocks)
                        )
                except Exception as e:
                    logger.warning(f"Error writing cache: {str(e)}")
            
            logger.info(f"Extracted text from {len(text_blocks)} pages")
            return '\n'.join(full_text), text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def highlight_contradictions(
        self,
        pdf_path: str,
        highlights: List[Dict[str, Any]],
        output_path: str,
        use_cache: bool = True
    ) -> None:
        """
        Highlight contradictions in a PDF file using simple text highlighting.
        
        Args:
            pdf_path: Path to the input PDF file
            highlights: List of text blocks to highlight
            output_path: Path to save the highlighted PDF
            use_cache: Whether to use cached results if available
        """
        if not highlights:
            logger.warning(f"No highlights provided for {pdf_path}")
            shutil.copy(pdf_path, output_path)
            return

        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(pdf_path, "highlighted")
            if cache_path.exists():
                try:
                    shutil.copy(cache_path, output_path)
                    logger.info(f"Using cached highlighted PDF: {cache_path}")
                    return
                except Exception as e:
                    logger.warning(f"Error using cache: {str(e)}")

        logger.info(f"Highlighting {len(highlights)} contradictions in {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Group highlights by page for better performance
            page_highlights = {}
            for highlight in highlights:
                text = highlight.get("text", "").strip()
                if not text:
                    continue
                    
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    matches = page.search_for(text)
                    if matches:
                        if page_num not in page_highlights:
                            page_highlights[page_num] = []
                        page_highlights[page_num].extend(matches)
            
            # Apply highlights page by page
            for page_num, matches in page_highlights.items():
                page = doc[page_num]
                for rect in matches:
                    page.add_highlight_annot(rect)

            # Save to both output and cache
            doc.save(output_path)
            if use_cache:
                doc.save(str(cache_path))
            doc.close()
            
            logger.info(f"Saved highlighted PDF to {output_path}")
            
        except Exception as e:
            logger.error(f"Error highlighting PDF: {str(e)}")
            raise
    
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
