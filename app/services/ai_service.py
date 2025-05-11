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

    def analyze_documents(self, main_text: str, target_text: str, main_blocks: List[Dict], target_blocks: List[Dict]) -> Dict:
        """
        Analyze two documents for contradictions and other insights.
        """
        logger.info("Starting document analysis")
        
        # Literal check for identical content
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

        # Get contradictions
        contradictions = self._extract_contradictions(main_text, target_text)
        logger.info(f"Found {len(contradictions)} potential contradictions")
        
        # Process contradictions for highlighting
        main_highlights = []
        target_highlights = []
        
        for contradiction in contradictions:
            main_highlights.append({"text": contradiction["main_text"]})
            target_highlights.append({"text": contradiction["target_text"]})
        
        # Get overall analysis
        similarity, risk_score, key_findings, missing_clauses = self._get_overall_analysis(main_text, target_text)
        
        return {
            "similarity": similarity,
            "contradictions": contradictions,
            "contradictory_segments": [c["main_text"] + " vs " + c["target_text"] for c in contradictions],
            "clause_analysis": [],
            "missing_clauses": missing_clauses,
            "overall_risk_score": risk_score,
            "key_findings": key_findings,
            "main_highlights": main_highlights,
            "target_highlights": target_highlights
        }

    def _extract_contradictions(self, main_text: str, target_text: str) -> List[Dict]:
        """
        Extract contradictions between two documents.
        """
        prompt = f"""
        You are an expert legal document analyst. Compare these two documents and identify specific contradictions.
        
        Focus solely on finding direct contradictions, such as:
        - Different dates, amounts, percentages, or numerical values
        - Different parties, addresses, or locations
        - Different obligations, rights, or terms that cannot both be true
        - Different conditions or requirements
        
        For each contradiction, provide:
        1. The exact text from the main document (keep it concise, just the contradictory phrase or sentence)
        2. The exact corresponding text from the target document
        3. A brief explanation of why they contradict
        
        Format your response as a JSON array of objects with these fields:
        - main_text: string containing the text from main document
        - target_text: string containing the text from target document
        - explanation: string explaining the contradiction
        
        DO NOT include any text other than the JSON array.
        
        Main Document:
        {main_text[:10000]}
        
        Target Document:
        {target_text[:10000]}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Find the first [ and last ] to extract just the JSON array
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                try:
                    contradictions = json.loads(json_str)
                    logger.info(f"Successfully parsed {len(contradictions)} contradictions")
                    return contradictions
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    return []
            else:
                logger.warning("Could not find JSON array in response")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting contradictions: {str(e)}")
            return []

    def _get_overall_analysis(self, main_text: str, target_text: str) -> Tuple[float, float, List[str], List[str]]:
        """
        Get overall analysis metrics between two documents.
        """
        prompt = f"""
        You are an expert legal document analyst. Compare these two documents and provide:
        
        1. A similarity percentage (0-100)
        2. A risk score (0-100) based on the significance of differences
        3. 3-5 key findings or recommendations
        4. A list of potentially missing clauses
        
        Format your response as JSON with these fields:
        - similarity: number (0-100)
        - risk_score: number (0-100)
        - key_findings: array of strings
        - missing_clauses: array of strings
        
        Main Document:
        {main_text[:5000]}
        
        Target Document:
        {target_text[:5000]}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            try:
                # Try to find and parse JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    data = json.loads(json_str)
                    return (
                        data.get("similarity", 50.0),
                        data.get("risk_score", 50.0),
                        data.get("key_findings", ["Analysis could not be completed"]),
                        data.get("missing_clauses", [])
                    )
            except:
                pass
                
            return 50.0, 50.0, ["Analysis could not be completed"], []
            
        except Exception as e:
            logger.error(f"Error getting overall analysis: {str(e)}")
            return 50.0, 50.0, ["Analysis could not be completed"], []