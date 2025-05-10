from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

class DocumentAnalysis(BaseModel):
    similarity: float = Field(description="Overall similarity percentage between documents")
    contradictions: List[str] = Field(description="List of contradictions found between documents")
    contradictory_segments: List[str] = Field(description="Specific text segments that contradict each other")

class AIService:
    def __init__(self):
        # Initialize Groq with callback manager
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="mixtral-8x7b-32768",
        callback_manager=callback_manager,
        verbose=True
        ) 
        self.parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)

    def analyze_documents(self, main_text: str, target_text: str) -> Dict:
        """
        Analyze two documents for similarities and contradictions using LangChain with Groq.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document analysis expert. Analyze the given documents for similarities and contradictions.
            {format_instructions}"""),
            ("user", """Analyze these two documents and provide:
            1. Overall similarity percentage
            2. List of contradictions found
            3. Specific segments that contradict each other

            Main Document:
            {main_text}

            Target Document:
            {target_text}""")
        ])

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )

        response = chain.invoke({
            "main_text": main_text,
            "target_text": target_text,
            "format_instructions": self.parser.get_format_instructions()
        })

        try:
            # Parse the response into our Pydantic model
            analysis = self.parser.parse(response['text'])
            return {
                "similarity": analysis.similarity,
                "contradictions": analysis.contradictions,
                "contradictory_segments": analysis.contradictory_segments
            }
        except Exception as e:
            # Fallback parsing if the structured output fails
            return self._fallback_parse(response['text'])

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