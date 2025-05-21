import os
import json
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

from .pdf_service import PDFService

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        self.pdf_service = PDFService()
        logger.info("AIService initialized with Gemini model")

    def process_pdf_documents(self, main_pdf_path: str, target_pdf_path: str, output_path: str) -> Dict:
        """
        Process two PDFs for comparison, handling scanned PDFs if necessary.
        
        Parameters:
            main_pdf_path (str): Path to the main PDF document
            target_pdf_path (str): Path to the target PDF document
            output_path (str): Directory where output files will be saved
            
        Returns:
            Dict: Analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Process main document
        main_searchable_path = os.path.join(output_path, "main_searchable.pdf")
        main_needs_ocr = self.pdf_service.check_if_scanned(main_pdf_path)
        
        if main_needs_ocr:
            logger.info(f"Main document appears to be scanned. Converting to searchable PDF.")
            self.pdf_service.convert_scanned_pdf(main_pdf_path, main_searchable_path)
            main_path_to_use = main_searchable_path
        else:
            main_path_to_use = main_pdf_path
            
        # Process target document
        target_searchable_path = os.path.join(output_path, "target_searchable.pdf")
        target_needs_ocr = self.pdf_service.check_if_scanned(target_pdf_path)
        
        if target_needs_ocr:
            logger.info(f"Target document appears to be scanned. Converting to searchable PDF.")
            self.pdf_service.convert_scanned_pdf(target_pdf_path, target_searchable_path)
            target_path_to_use = target_searchable_path
        else:
            target_path_to_use = target_pdf_path
        
        # Extract text from processed PDFs
        main_text, main_blocks = self.pdf_service.extract_text_from_pdf(main_path_to_use)
        target_text, target_blocks = self.pdf_service.extract_text_from_pdf(target_path_to_use)
        
        # Analyze documents
        analysis_result = self.analyze_documents(main_text, target_text, main_blocks, target_blocks)
        
        # Create highlighted versions
        main_highlighted_path = os.path.join(output_path, "main_highlighted.pdf")
        target_highlighted_path = os.path.join(output_path, "target_highlighted.pdf")
        
        self.pdf_service.highlight_contradictions(
            main_path_to_use, 
            analysis_result.get("main_highlights", []), 
            main_highlighted_path
        )
        
        self.pdf_service.highlight_contradictions(
            target_path_to_use,
            analysis_result.get("target_highlights", []),
            target_highlighted_path
        )
        
        # Add paths to the result
        analysis_result["main_highlighted_pdf"] = main_highlighted_path
        analysis_result["target_highlighted_pdf"] = target_highlighted_path
        
        # For display purposes, also add paths to the searchable PDFs 
        analysis_result["main_searchable_pdf"] = main_path_to_use
        analysis_result["target_searchable_pdf"] = target_path_to_use
        
        return analysis_result
    
    def analyze_documents(
        self, 
        main_text: str, 
        target_text: str, 
        main_blocks: List[Dict], 
        target_blocks: List[Dict]
    ) -> Dict:
        logger.info("Starting document analysis")

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

        contradictions = self._extract_contradictions(main_text, target_text)
        logger.info(f"Found {len(contradictions)} contradictions or inconsistencies")

        main_highlights = [{"text": c.get("main_text", "")} for c in contradictions]
        target_highlights = [{"text": c.get("target_text", "")} for c in contradictions]

        similarity, risk_score, key_findings, missing_clauses = self._get_overall_analysis(main_text, target_text)

        return {
            "similarity": similarity,
            "contradictions": contradictions,
            "contradictory_segments": [
                f"{c.get('main_text', '')} vs {c.get('target_text', '')}" for c in contradictions
            ],
            "clause_analysis": [],
            "missing_clauses": missing_clauses,
            "overall_risk_score": risk_score,
            "key_findings": key_findings,
            "main_highlights": main_highlights,
            "target_highlights": target_highlights
        }

    def _extract_contradictions(self, main_text: str, target_text: str) -> List[Dict]:
        prompt = f"""
        You are a legal contract reviewer and AI trained to analyze and compare legal documents thoroughly.

        Compare the two documents and identify ALL critical contradictions, discrepancies, or semantic differences that could lead to misunderstanding, legal risk, or breach. Consider both explicit and subtle inconsistencies.

        Key areas to evaluate include but are not limited to:
        - Delivery terms, deadlines, ETA
        - Functional and non-functional requirements
        - Responsibilities, roles, obligations of each party
        - Payment terms, penalties, bonuses
        - Timeline and project durations
        - Evaluation criteria or acceptance standards
        - Confidentiality, IP rights, security clauses
        - Dispute resolution or termination procedures
        - Jurisdiction or governing law
        - Scope creep prevention, change control mechanisms
        - Definitions and terminology differences
        - Cross-references or clause dependencies
        - Any clause that appears in one doc but not the other
        - Reworded clauses with potentially altered intent

        Return a JSON array where each item includes:
        - main_text: Excerpt from main document (or "MISSING" if absent)
        - target_text: Corresponding excerpt from target document (or "MISSING")
        - explanation: Short explanation of why this difference matters

        Format your response as a raw JSON array only, with no surrounding explanation.

        Main Document:
        {main_text[:10000]}

        Target Document:
        {target_text[:10000]}
        """

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

    def _get_overall_analysis(self, main_text: str, target_text: str) -> Tuple[float, float, List[str], List[str]]:
        prompt = f"""
        You are an advanced legal analyst AI.

        Compare the following two contract documents and return a structured assessment of overall risk and similarities.

        Respond strictly with a JSON object containing:
        {{
          "similarity": number (0–100),  // Overall textual and semantic similarity
          "risk_score": number (0–100),  // Risk posed by differences in clauses
          "key_findings": [string],      // 3-5 insightful findings or risks
          "missing_clauses": [string]    // Important clauses missing in either document
        }}

        Do not include explanatory text. Return only valid JSON.

        Main Document:
        {main_text[:5000]}

        Target Document:
        {target_text[:5000]}
        """

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