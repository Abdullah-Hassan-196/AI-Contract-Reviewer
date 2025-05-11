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

        # First, get structured contradiction data
        contradictions = self._extract_contradictions(main_text, target_text)
        logger.info(f"Found {len(contradictions)} potential contradictions")
        
        # Then process each contradiction to find the exact locations in both documents
        processed_contradictions = []
        main_highlights = []
        target_highlights = []
        
        for contradiction in contradictions:
            # Find matching blocks in both documents
            main_matches = self.pdf_service.find_text_in_blocks(contradiction["main_text"], main_blocks)
            target_matches = self.pdf_service.find_text_in_blocks(contradiction["target_text"], target_blocks)
            
            if main_matches and target_matches:
                # Use the first match for simplicity (you could enhance this to find the best match)
                main_match = main_matches[0]
                target_match = target_matches[0]
                
                # Create highlight objects
                main_highlight = {
                    "page": main_match["page"],
                    "bbox": main_match["bbox"],
                    "text": main_match["text"]
                }
                
                target_highlight = {
                    "page": target_match["page"],
                    "bbox": target_match["bbox"],
                    "text": target_match["text"]
                }
                
                # Add to our lists
                main_highlights.append(main_highlight)
                target_highlights.append(target_highlight)
                
                # Add to processed contradictions
                processed_contradictions.append({
                    "main_text": contradiction["main_text"],
                    "target_text": contradiction["target_text"],
                    "explanation": contradiction["explanation"],
                    "main_highlight": main_highlight,
                    "target_highlight": target_highlight
                })
                
                logger.info(f"Processed contradiction: '{contradiction['main_text'][:30]}...' vs '{contradiction['target_text'][:30]}...'")
            else:
                logger.warning(f"Could not find text blocks for contradiction: {contradiction['main_text'][:30]}...")
        
        # Get overall analysis
        similarity, risk_score, key_findings, missing_clauses = self._get_overall_analysis(main_text, target_text)
        
        return {
            "similarity": similarity,
            "contradictions": processed_contradictions,
            "contradictory_segments": [c["main_text"] + " vs " + c["target_text"] for c in contradictions],
            "clause_analysis": [],  # This would need more complex analysis
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
        {main_text[:10000]}  # Limiting to 10k chars to avoid token limits
        
        Target Document:
        {target_text[:10000]}  # Limiting to 10k chars to avoid token limits
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
                    logger.debug(f"Raw JSON string: {json_str[:100]}...")
                    return self._fallback_extract_contradictions(response_text)
            else:
                logger.warning("Could not find JSON array in response")
                return self._fallback_extract_contradictions(response_text)
                
        except Exception as e:
            logger.error(f"Error extracting contradictions: {str(e)}")
            return []

    def _fallback_extract_contradictions(self, response_text: str) -> List[Dict]:
        """
        Fallback method to extract contradictions if JSON parsing fails.
        """
        logger.info("Using fallback contradiction extraction")
        contradictions = []
        
        # Look for patterns that might indicate contradictions
        lines = response_text.split('\n')
        current_contradiction = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "main" in line.lower() and ":" in line:
                # Start a new contradiction if we were tracking one
                if current_contradiction and "main_text" in current_contradiction and "target_text" in current_contradiction:
                    if "explanation" not in current_contradiction:
                        current_contradiction["explanation"] = "Potential contradiction identified"
                    contradictions.append(current_contradiction)
                    current_contradiction = {}
                
                current_contradiction["main_text"] = line.split(":", 1)[1].strip()
            elif "target" in line.lower() and ":" in line:
                current_contradiction["target_text"] = line.split(":", 1)[1].strip()
            elif "explanation" in line.lower() and ":" in line:
                current_contradiction["explanation"] = line.split(":", 1)[1].strip()
                
            # If we have all fields, add the contradiction
            if current_contradiction and "main_text" in current_contradiction and "target_text" in current_contradiction:
                if "explanation" not in current_contradiction:
                    current_contradiction["explanation"] = "Potential contradiction identified"
                contradictions.append(current_contradiction)
                current_contradiction = {}
        
        # Add the last contradiction if there is one
        if current_contradiction and "main_text" in current_contradiction and "target_text" in current_contradiction:
            if "explanation" not in current_contradiction:
                current_contradiction["explanation"] = "Potential contradiction identified"
            contradictions.append(current_contradiction)
        
        logger.info(f"Fallback extraction found {len(contradictions)} contradictions")
        return contradictions

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
        {main_text[:5000]}  # Limiting to 5k chars to avoid token limits
        
        Target Document:
        {target_text[:5000]}  # Limiting to 5k chars to avoid token limits
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
                
            # Fallback to simple parsing
            similarity = 50.0  # Default
            risk_score = 50.0  # Default
            key_findings = ["Analysis could not be completed"]
            missing_clauses = []
            
            for line in response_text.split('\n'):
                line = line.strip()
                if "similarity" in line.lower() and ":" in line:
                    try:
                        similarity = float(line.split(":")[1].strip().replace("%", ""))
                    except:
                        pass
                elif "risk" in line.lower() and "score" in line.lower() and ":" in line:
                    try:
                        risk_score = float(line.split(":")[1].strip().replace("%", ""))
                    except:
                        pass
                elif "finding" in line.lower() and ":" in line:
                    key_findings = [line.split(":", 1)[1].strip()]
                elif "missing" in line.lower() and "clause" in line.lower() and ":" in line:
                    missing_clauses.append(line.split(":", 1)[1].strip())
            
            return similarity, risk_score, key_findings, missing_clauses
            
        except Exception as e:
            logger.error(f"Error getting overall analysis: {str(e)}")
            return 50.0, 50.0, ["Analysis could not be completed"], []