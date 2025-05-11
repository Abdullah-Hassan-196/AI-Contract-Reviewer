from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class Highlight(BaseModel):
    page: int = Field(description="Page number where the highlight appears")
    bbox: List[float] = Field(description="Bounding box coordinates [x0, y0, x1, y1]")
    text: str = Field(description="The highlighted text")

class ClauseAnalysis(BaseModel):
    clause_text: str = Field(description="The text of the clause")
    risk_level: str = Field(description="Risk level: Low, Medium, High")
    risk_explanation: str = Field(description="Explanation of the risk assessment")
    legal_implications: List[str] = Field(description="List of legal implications")
    recommendations: List[str] = Field(description="List of recommendations")
    industry_standards: List[str] = Field(description="Relevant industry standards")
    highlights: List[Highlight] = Field(description="Locations of this clause in the document")

class DocumentAnalysis(BaseModel):
    similarity: float = Field(description="Overall similarity percentage between documents")
    contradictions: List[str] = Field(description="List of contradictions found between documents")
    contradictory_segments: List[str] = Field(description="Specific text segments that contradict each other")
    clause_analysis: List[ClauseAnalysis] = Field(description="Analysis of each clause")
    missing_clauses: List[str] = Field(description="List of potentially missing clauses")
    overall_risk_score: float = Field(description="Overall risk score (0-100)")
    key_findings: List[str] = Field(description="Key findings and recommendations")
    main_highlights: List[Highlight] = Field(description="Highlights for the main document")
    target_highlights: List[Highlight] = Field(description="Highlights for the target document")

class AIService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

    def analyze_documents(self, main_text: str, target_text: str, main_blocks: List[dict], target_blocks: List[dict]) -> Dict:
        # Literal check for identical content
        if main_text.strip() == target_text.strip():
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

        prompt = f"""
You are an expert legal document analyst. Analyze the given documents for:
1. Similarities and contradictions
2. Risk assessment for each clause
3. Legal implications
4. Industry standard compliance
5. Missing clauses
6. Overall risk assessment

When searching for contradictions, pay special attention to:
- Differences in numerical values (amounts, percentages, dates, etc.)
- Changes in locations (place names, addresses, etc.)
- Sentences that are in direct conflict or create obligations that cannot both be true

If the content of the two documents is 100% the same (regardless of file name), return a similarity score of 100% and no contradictions.

For each contradiction you find, output:
- The exact sentence(s) from the Main Document
- The exact sentence(s) from the Target Document
- A brief explanation of the contradiction

Format the 'Contradictions Found' section as a list of objects, each containing:
- main_sentence: The sentence from the Main Document
- target_sentence: The sentence from the Target Document
- explanation: A brief explanation of the contradiction

For each contradiction or important clause, provide the exact location in the document using the provided text blocks.

Main Document:
{main_text}

Target Document:
{target_text}

Main Document Blocks: {main_blocks}
Target Document Blocks: {target_blocks}
"""

        response = self.model.generate_content(prompt)
        # You will need to parse the response.text into your expected output format.
        # (You may want to use a stricter output format or JSON mode if Gemini supports it.)

        # For now, fallback to your existing _fallback_parse logic:
        return self._fallback_parse(response.text)

    def _fallback_parse(self, response: str) -> Dict:
        """
        Fallback method to parse the response if the structured output fails.
        """
        logger.info("Using fallback parsing...")
        lines = response.split('\n')
        result = {
            "similarity": 0.0,
            "contradictions": [],
            "contradictory_segments": [],
            "clause_analysis": [],
            "missing_clauses": [],
            "overall_risk_score": 0.0,
            "key_findings": [],
            "main_highlights": [],
            "target_highlights": []
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "similarity" in line.lower():
                try:
                    result["similarity"] = float(line.split(":")[1].strip().replace("%", ""))
                except:
                    pass
            elif "contradiction" in line.lower():
                result["contradictions"].append(line.split(":")[1].strip())
            elif "segment" in line.lower():
                result["contradictory_segments"].append(line.split(":")[1].strip())
            elif "risk score" in line.lower():
                try:
                    result["overall_risk_score"] = float(line.split(":")[1].strip().replace("%", ""))
                except:
                    pass
            elif "finding" in line.lower():
                result["key_findings"].append(line.split(":")[1].strip())
            elif "missing clause" in line.lower():
                result["missing_clauses"].append(line.split(":")[1].strip())

        return result 