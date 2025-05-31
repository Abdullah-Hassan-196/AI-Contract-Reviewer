import os
import json
import logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from functools import lru_cache
import hashlib
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .pdf_service import PDFService

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self):
        """Initialize AI service with model and cache settings"""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        self.pdf_service = PDFService()
        self._cache_dir = Path("temp/ai_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=4)
        logger.info("AIService initialized with Gemini model")
    
    def _get_cache_path(self, text1: str, text2: str, operation: str) -> Path:
        """Generate cache path for an analysis operation"""
        combined = f"{text1[:1000]}{text2[:1000]}"
        file_hash = hashlib.md5(combined.encode()).hexdigest()
        return self._cache_dir / f"{operation}_{file_hash}.json"
    
    @lru_cache(maxsize=100)
    def _get_text_hash(self, text: str) -> str:
        """Get hash of text for caching"""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def analyze_documents(
        self, 
        main_text: str, 
        target_text: str, 
        main_blocks: List[Dict], 
        target_blocks: List[Dict],
        use_cache: bool = True
    ) -> Dict:
        """
        Analyze two documents for contradictions and differences.
        
        Args:
            main_text: Text from main document
            target_text: Text from target document
            main_blocks: Text blocks from main document
            target_blocks: Text blocks from target document
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting document analysis")
        
        # Check if documents are identical
        if main_text.strip() == target_text.strip():
            logger.info("Documents are identical")
            return {
                "similarity": 100.0,
                "contradictions": [],
                "contradictory_segments": [],
                "clause_analysis": [],
                "missing_clauses": [],
                "overall_risk_score": 0.0,
                "key_findings": ["The documents are identical."],
                "main_highlights": [],
                "target_highlights": []
            }
        
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(main_text, target_text, "analysis")
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                        logger.info("Using cached analysis results")
                        return cached_data
                except Exception as e:
                    logger.warning(f"Error reading cache: {str(e)}")
        
        try:
            # Run analysis tasks concurrently
            contradictions_task = asyncio.create_task(
                asyncio.to_thread(
                    self._extract_contradictions,
                    main_text,
                    target_text
                )
            )
            analysis_task = asyncio.create_task(
                asyncio.to_thread(
                    self._get_overall_analysis,
                    main_text,
                    target_text
                )
            )
            
            contradictions, analysis_result = await asyncio.gather(
                contradictions_task, analysis_task
            )
            similarity, risk_score, key_findings, missing_clauses = analysis_result
            
            logger.info(
                f"Found {len(contradictions)} contradictions or inconsistencies"
            )
            
            main_highlights = [{"text": c.get("main_text", "")} for c in contradictions]
            target_highlights = [{"text": c.get("target_text", "")} for c in contradictions]
            
            result = {
                "similarity": similarity,
                "contradictions": contradictions,
                "contradictory_segments": [
                    f"{c.get('main_text', '')} vs {c.get('target_text', '')}" 
                    for c in contradictions
                ],
                "clause_analysis": [],
                "missing_clauses": missing_clauses,
                "overall_risk_score": risk_score,
                "key_findings": key_findings,
                "main_highlights": main_highlights,
                "target_highlights": target_highlights
            }
            
            # Cache results
            if use_cache:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(result, f)
                except Exception as e:
                    logger.warning(f"Error writing cache: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            raise
    
    def _extract_contradictions(
        self,
        main_text: str,
        target_text: str,
        max_length: int = 10000
    ) -> List[Dict]:
        """
        Extract contradictions between two documents.
        
        Args:
            main_text: Text from main document
            target_text: Text from target document
            max_length: Maximum text length to process
            
        Returns:
            List of contradiction dictionaries
        """
        prompt = (
            "You are a legal contract reviewer and AI trained to analyze and "
            "compare legal documents thoroughly.\n\n"
            "Compare the two documents and identify ALL critical contradictions, "
            "discrepancies, or semantic differences that could lead to "
            "misunderstanding, legal risk, or breach. Consider both explicit and "
            "subtle inconsistencies.\n\n"
            "Key areas to evaluate include but are not limited to:\n"
            "- Delivery terms, deadlines, ETA\n"
            "- Functional and non-functional requirements\n"
            "- Responsibilities, roles, obligations of each party\n"
            "- Payment terms, penalties, bonuses\n"
            "- Timeline and project durations\n"
            "- Evaluation criteria or acceptance standards\n"
            "- Confidentiality, IP rights, security clauses\n"
            "- Dispute resolution or termination procedures\n"
            "- Jurisdiction or governing law\n"
            "- Scope creep prevention, change control mechanisms\n"
            "- Definitions and terminology differences\n"
            "- Cross-references or clause dependencies\n"
            "- Any clause that appears in one doc but not the other\n"
            "- Reworded clauses with potentially altered intent\n\n"
            "Return a JSON array where each item includes:\n"
            "- main_text: Excerpt from main document (or \"MISSING\" if absent)\n"
            "- target_text: Corresponding excerpt from target document (or \"MISSING\")\n"
            "- explanation: Short explanation of why this difference matters\n\n"
            "Format your response as a raw JSON array only, with no surrounding "
            "explanation.\n\n"
            f"Main Document:\n{main_text[:max_length]}\n\n"
            f"Target Document:\n{target_text[:max_length]}"
        )

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1

            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON array detected")
                return []

            return json.loads(response_text[start_idx:end_idx])

        except Exception as e:
            logger.error(f"Model error: {e}")
            return []

    def _get_overall_analysis(
        self,
        main_text: str,
        target_text: str,
        max_length: int = 5000
    ) -> Tuple[float, float, List[str], List[str]]:
        """
        Get overall analysis of document differences.
        
        Args:
            main_text: Text from main document
            target_text: Text from target document
            max_length: Maximum text length to process
            
        Returns:
            Tuple of (similarity, risk_score, key_findings, missing_clauses)
        """
        prompt = (
            "You are an advanced legal analyst AI.\n\n"
            "Compare the following two contract documents and return a structured "
            "assessment of overall risk and similarities.\n\n"
            "Respond strictly with a JSON object containing:\n"
            "{\n"
            '  "similarity": number (0–100),  // Overall textual and semantic '
            "similarity\n"
            '  "risk_score": number (0–100),  // Risk posed by differences in '
            "clauses\n"
            '  "key_findings": [string],      // 3-5 insightful findings or risks\n'
            '  "missing_clauses": [string]    // Important clauses missing in '
            "either document\n"
            "}\n\n"
            "Do not include explanatory text. Return only valid JSON.\n\n"
            f"Main Document:\n{main_text[:max_length]}\n\n"
            f"Target Document:\n{target_text[:max_length]}"
        )

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON object detected in analysis response")
                return 50.0, 50.0, ["Unable to extract findings"], []

            data = json.loads(response_text[start_idx:end_idx])
            return (
                data.get("similarity", 50.0),
                data.get("risk_score", 50.0),
                data.get("key_findings", ["No findings"]),
                data.get("missing_clauses", [])
            )

        except Exception as e:
            logger.error(f"Error in overall analysis: {e}")
            return 50.0, 50.0, ["Analysis failed"], []
    
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
