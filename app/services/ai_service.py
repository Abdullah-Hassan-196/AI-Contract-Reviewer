import os
from typing import List, Dict, Any, Tuple
import logging
import json
from dotenv import load_dotenv
import google.generativeai as genai

from .pdf_service import PDFService

# Configure logging
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

    def analyze_documents(
        self, 
        main_text: str, 
        target_text: str, 
        main_blocks: List[Dict], 
        target_blocks: List[Dict]
    ) -> Dict:
        """
        Analyze two documents for contradictions and insights.
        """
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
        """
        Identify all contradictions and inconsistencies critical to contract integrity.
        """
        prompt = f"""
        You are an expert contract reviewer and legal document analyst. Carefully analyze and compare the following two documents.

        Your goal is to detect and list ALL contradictions, inconsistencies, or missing information that are legally or functionally significant in a contractual context.

        Focus on, but do not limit yourself to, the following areas:
        - Delivery terms and Estimated Time of Arrival (ETA)
        - Functional and non-functional requirements
        - Roles, responsibilities, and obligations of each party
        - Deadlines, durations, and timelines
        - Evaluation metrics and performance standards
        - Payment terms, percentages, penalties, or bonuses
        - Clauses that exist in one document but are completely missing in the other
        - Any clause that is rephrased in a way that changes its meaning
        - Definitions or critical terms being added, removed, or altered
        - Security, confidentiality, and intellectual property rights
        - Dispute resolution or termination clauses

        Output a JSON array of contradictions. Each object should include:
        - main_text: a concise excerpt from the main document (or "MISSING" if not present)
        - target_text: a concise excerpt from the target document (or "MISSING" if not present)
        - explanation: a brief but clear reason why this represents a contradiction or legal inconsistency

        Return ONLY the JSON array. Do NOT include introductory or explanatory text outside the JSON.

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

            json_str = response_text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                return []

        except Exception as e:
            logger.error(f"Model error: {e}")
            return []

    def _get_overall_analysis(self, main_text: str, target_text: str) -> Tuple[float, float, List[str], List[str]]:
        """
        Get similarity score, risk, findings, and missing clauses.
        """
        prompt = f"""
        You are an expert legal analyst. Review and compare these two contract documents.

        Return a JSON object that includes:
        - similarity: overall similarity percentage (0-100)
        - risk_score: overall risk score (0-100), based on the importance of differences
        - key_findings: 3–5 brief, important findings or concerns
        - missing_clauses: a list of important clauses that exist in one document but are absent in the other

        Output format:
        {{
          "similarity": number,
          "risk_score": number,
          "key_findings": [string],
          "missing_clauses": [string]
        }}

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

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            return (
                data.get("similarity", 50.0),
                data.get("risk_score", 50.0),
                data.get("key_findings", ["No findings"]),
                data.get("missing_clauses", [])
            )

        except Exception as e:
            logger.error(f"Error in overall analysis: {e}")
            return 50.0, 50.0, ["Analysis failed"], []
