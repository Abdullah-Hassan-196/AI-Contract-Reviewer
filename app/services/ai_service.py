from typing import List, Dict
import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()

class DocumentAnalysis(BaseModel):
    similarity: float = Field(description="Overall similarity percentage between documents")
    contradictions: List[str] = Field(description="List of contradictions found between documents")
    contradictory_segments: List[str] = Field(description="Specific text segments that contradict each other")

class AIService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"

    def analyze_documents(self, main_text: str, target_text: str) -> Dict:
        """
        Analyze two documents for similarities and contradictions using Groq API directly.
        """
        prompt = f"""You are a document analysis expert. Analyze these two documents and provide a JSON response with:
1. Overall similarity percentage (as a float between 0 and 100)
2. List of contradictions found
3. Specific segments that contradict each other

Main Document:
{main_text}

Target Document:
{target_text}

Provide the response in this exact JSON format:
{{
    "similarity": <float>,
    "contradictions": [<list of strings>],
    "contradictory_segments": [<list of strings>]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a document analysis expert that provides responses in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            # Extract the JSON response from the completion
            response_text = response.choices[0].message.content
            try:
                # Try to parse as DocumentAnalysis
                analysis = DocumentAnalysis.parse_raw(response_text)
                return {
                    "similarity": analysis.similarity,
                    "contradictions": analysis.contradictions,
                    "contradictory_segments": analysis.contradictory_segments
                }
            except Exception as e:
                # Fallback parsing if the structured output fails
                return self._fallback_parse(response_text)
                
        except Exception as e:
            raise Exception(f"Error analyzing documents: {str(e)}")

    def _fallback_parse(self, response: str) -> Dict:
        """
        Fallback method to parse the response if the structured output fails.
        """
        lines = response.split('\n')
        result = {
            "similarity": 0.0,
            "contradictions": [],
            "contradictory_segments": []
        }

        for line in lines:
            if "similarity" in line.lower():
                try:
                    result["similarity"] = float(line.split(":")[1].strip().replace("%", ""))
                except:
                    pass
            elif "contradiction" in line.lower():
                result["contradictions"].append(line.split(":")[1].strip())
            elif "segment" in line.lower():
                result["contradictory_segments"].append(line.split(":")[1].strip())

        return result 