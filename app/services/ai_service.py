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
        # Initialize Groq with callback manager
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        logger.info("Initializing Groq client...")
        try:
            self.llm = ChatGroq(
                api_key=api_key,
                model="llama-3.3-70b-versatile",
                callback_manager=callback_manager,
                verbose=True
            )
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}")
            raise
        
        self.parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)

    def analyze_documents(self, main_text: str, target_text: str, main_blocks: List[dict], target_blocks: List[dict]) -> Dict:
        """
        Analyze two documents for similarities, contradictions, and legal implications using LangChain with Groq.
        """
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
        
        logger.info("Starting document analysis...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
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
{format_instructions}
"""),
            ("user", """
Analyze these two documents comprehensively:

Main Document:
{main_text}

Target Document:
{target_text}

Consider:
- Legal precedents
- Industry best practices
- Potential risks and liabilities
- Missing or problematic clauses
- Recommendations for improvement

When identifying contradictions, focus on:
- Numerical differences (amounts, dates, percentages)
- Location changes (addresses, place names)
- Sentences that are in direct conflict

For each contradiction, return the exact sentences from both documents and a brief explanation.
Main Document Blocks: {main_blocks}
Target Document Blocks: {target_blocks}
""")
        ])

        try:
            logger.info("Creating analysis chain...")
            chain = prompt | self.llm | self.parser

            logger.info("Invoking analysis chain...")
            response = chain.invoke({
                "main_text": main_text,
                "target_text": target_text,
                "main_blocks": main_blocks,
                "target_blocks": target_blocks,
                "format_instructions": self.parser.get_format_instructions()
            })

            logger.info("Analysis completed successfully")
            return {
                "similarity": response.similarity,
                "contradictions": response.contradictions,
                "contradictory_segments": response.contradictory_segments,
                "clause_analysis": [clause.dict() for clause in response.clause_analysis],
                "missing_clauses": response.missing_clauses,
                "overall_risk_score": response.overall_risk_score,
                "key_findings": response.key_findings,
                "main_highlights": [highlight.dict() for highlight in response.main_highlights],
                "target_highlights": [highlight.dict() for highlight in response.target_highlights]
            }
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            return self._fallback_parse(str(e))

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