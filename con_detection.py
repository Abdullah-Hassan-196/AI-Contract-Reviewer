import nltk
# Download all required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import PyPDF2
import re
import spacy
from sentence_transformers import SentenceTransformer
import textdistance
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nest_asyncio
import logging
import json
from time import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize
nest_asyncio.apply()

# Create app
app = FastAPI(title="Enhanced Document Matching and Contradiction Detection")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
nlp = None  # spaCy model
sentence_model = None  # Sentence transformer model
nli_tokenizer = None  # NLI tokenizer
nli_model = None  # NLI model
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_models():
    global nlp, sentence_model, nli_tokenizer, nli_model

    logger.info("Loading NLP models...")
    start_time = time()

    # Load spaCy
    if nlp is None:
        try:
            nlp = spacy.load('en_core_web_md')
            logger.info("Loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            nlp = None

    # Load Sentence Transformer
    if sentence_model is None:
        try:
            sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            logger.info("Loaded Sentence Transformer model")
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer model: {str(e)}")
            sentence_model = None

    # Replace your NLI model loading code with:
    if nli_tokenizer is None or nli_model is None:
        try:
            MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"  # More powerful model
            nli_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            nli_model.eval()
            if torch.cuda.is_available():
                nli_model = nli_model.cuda()  # Move to GPU if available
            logger.info("Loaded NLI model")
        except Exception as e:
            logger.error(f"Error loading NLI model: {str(e)}")
            # Fallback to smaller model
            try:
                MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
                nli_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                nli_model.eval()
                logger.info("Loaded fallback NLI model")
            except Exception as e:
                logger.error(f"Error loading fallback NLI model: {str(e)}")
                nli_tokenizer = None
                nli_model = None

# Response models
class SimilarityResponse(BaseModel):
    overall_similarity: float
    contradictions: List[Dict[str, Any]]
    similarity_scores: List[Dict[str, Any]]
    main_document_text: str
    target_document_text: str
    main_segments: List[str]
    target_segments: List[str]
    semantic_similarity: float
    exact_matches: List[Dict[str, Any]]
    analysis_summary: Dict[str, Any]

class DocumentSegment(BaseModel):
    text: str
    index: int
    key_concepts: Optional[List[str]] = None
    named_entities: Optional[List[Dict[str, str]]] = None

# Text processing functions
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(400, f"Could not extract text from PDF: {str(e)}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise HTTPException(400, f"Could not extract text from DOCX: {str(e)}")
    return text

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {str(e)}")
            raise HTTPException(400, f"Could not extract text from TXT: {str(e)}")

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    raise HTTPException(400, f"Unsupported file format: {ext}")

def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Replace multiple spaces, newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters but keep periods for sentence tokenization
    text = re.sub(r'[^\w\s\.]', ' ', text)
    return text

def clean_segment(segment):
    """Clean and normalize a text segment"""
    if not segment:
        return ""
    # Lowercase
    segment = segment.lower()
    # Tokenize words
    words = word_tokenize(segment)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

def segment_text(text, min_words=3):
    """Split text into sentences and filter short ones"""
    segments = []
    for s in sent_tokenize(text):
        s = s.strip()
        if s and len(s.split()) >= min_words:
            segments.append(s)
    return segments

def extract_key_concepts(segment):
    """Extract key concepts from a text segment using spaCy"""
    if not nlp:
        return []

    try:
        doc = nlp(segment)
        # Extract noun chunks and named entities as key concepts
        concepts = set()

        # Add noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 1:
                concepts.add(chunk.text.lower())

        # Add named entities
        for ent in doc.ents:
            concepts.add(ent.text.lower())

        return list(concepts)
    except Exception as e:
        logger.warning(f"Error extracting concepts: {str(e)}")
        return []

def extract_named_entities(segment):
    """Extract named entities from a text segment"""
    if not nlp:
        return []

    try:
        doc = nlp(segment)
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "type": ent.label_})
        return entities
    except Exception as e:
        logger.warning(f"Error extracting entities: {str(e)}")
        return []

def enrich_segments(segments):
    """Add key concepts and named entities to segments"""
    enriched = []
    for i, segment in enumerate(segments):
        concepts = extract_key_concepts(segment)
        entities = extract_named_entities(segment)
        enriched.append({
            "text": segment,
            "index": i,
            "key_concepts": concepts,
            "named_entities": entities
        })
    return enriched

def calculate_tfidf_similarity(text1, text2):
    """Calculate TF-IDF based cosine similarity"""
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception as e:
        logger.warning(f"Error in TF-IDF similarity: {str(e)}")
        return 0.0

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using Sentence Transformers"""
    if not sentence_model:
        return 0.0

    try:
        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = sentence_model.encode([text1, text2])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    except Exception as e:
        logger.warning(f"Error in semantic similarity: {str(e)}")
        return 0.0

def calculate_fuzzy_similarity(text1, text2):
    """Calculate fuzzy token similarity"""
    try:
        return fuzz.token_sort_ratio(text1, text2) / 100.0
    except Exception as e:
        logger.warning(f"Error in fuzzy similarity: {str(e)}")
        return 0.0

def calculate_jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between segments"""
    try:
        words1 = set(clean_segment(text1).split())
        words2 = set(clean_segment(text2).split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)
    except Exception as e:
        logger.warning(f"Error in Jaccard similarity: {str(e)}")
        return 0.0

def get_segment_similarity(main_segs, target_segs, threshold=0.5):
    """Find the most similar pairs of segments above threshold"""
    if not main_segs or not target_segs:
        return []

    results = []

    # For each main segment, find the best match in target segments
    for i, main_seg in enumerate(main_segs):
        best = {"score": 0.0, "main_segment": main_seg, "target_segment": "", "main_index": i, "target_index": -1}

        for j, target_seg in enumerate(target_segs):
            # Calculate multiple similarity metrics
            tfidf_sim = calculate_tfidf_similarity(main_seg, target_seg)
            semantic_sim = calculate_semantic_similarity(main_seg, target_seg)
            fuzzy_sim = calculate_fuzzy_similarity(main_seg, target_seg)
            jaccard_sim = calculate_jaccard_similarity(main_seg, target_seg)

            # Weighted average of similarity scores
            score = (0.3 * tfidf_sim + 0.4 * semantic_sim + 0.2 * fuzzy_sim + 0.1 * jaccard_sim)

            if score > best["score"]:
                best = {
                    "score": score,
                    "main_segment": main_seg,
                    "target_segment": target_seg,
                    "main_index": i,
                    "target_index": j,
                    "metrics": {
                        "tfidf": tfidf_sim,
                        "semantic": semantic_sim,
                        "fuzzy": fuzzy_sim,
                        "jaccard": jaccard_sim
                    }
                }

        if best["score"] > threshold:
            results.append(best)

    # Sort by similarity score
    return sorted(results, key=lambda x: x["score"], reverse=True)

def find_exact_matches(main_segs, target_segs):
    """Find segments that are exactly or nearly identical"""
    exact_matches = []

    for i, main_seg in enumerate(main_segs):
        for j, target_seg in enumerate(target_segs):
            # Normalize for comparison
            main_clean = re.sub(r'\s+', ' ', main_seg.lower().strip())
            target_clean = re.sub(r'\s+', ' ', target_seg.lower().strip())

            # Calculate edit distance ratio
            edit_distance = textdistance.levenshtein.normalized_similarity(main_clean, target_clean)

            # Consider it a match if very similar
            if edit_distance > 0.9:
                exact_matches.append({
                    "main_segment": main_seg,
                    "target_segment": target_seg,
                    "main_index": i,
                    "target_index": j,
                    "similarity": edit_distance
                })

    return exact_matches

def detect_contradictions(main_segs, target_segs):
    """Enhanced contradiction detection"""
    if not nli_tokenizer or not nli_model:
        return []

    contradictions = []

    # Compare all possible pairs (not just similar ones)
    for i, main_seg in enumerate(main_segs[:50]):  # Limit to first 50 for performance
        for j, target_seg in enumerate(target_segs[:50]):
            # Skip very short segments
            if len(main_seg.split()) < 5 or len(target_seg.split()) < 5:
                continue

            try:
                # Prepare inputs with better formatting
                premise = main_seg[:512]  # Truncate to model max length
                hypothesis = target_seg[:512]

                inputs = nli_tokenizer(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )

                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Get predictions
                with torch.no_grad():
                    outputs = nli_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).squeeze(0)

                # Get contradiction probability
                contradiction_score = float(probs[2].item()) if len(probs) > 2 else 0.0

                # Lower threshold to 0.5 and require some concept overlap
                main_concepts = extract_key_concepts(main_seg)
                target_concepts = extract_key_concepts(target_seg)
                common_concepts = set(main_concepts).intersection(set(target_concepts))

                if contradiction_score > 0.5 and len(common_concepts) > 0:
                    contradictions.append({
                        "main_segment": main_seg,
                        "target_segment": target_seg,
                        "main_index": i,
                        "target_index": j,
                        "contradiction_score": contradiction_score,
                        "common_concepts": list(common_concepts)
                    })

            except Exception as e:
                logger.warning(f"Error in contradiction detection: {str(e)}")
                continue

    return sorted(contradictions, key=lambda x: x["contradiction_score"], reverse=True)

def generate_analysis_summary(main_text, target_text, similarity, contradictions, exact_matches):
    """Generate a summary of the analysis results"""
    main_word_count = len(main_text.split())
    target_word_count = len(target_text.split())

    summary = {
        "main_document": {
            "word_count": main_word_count,
            "sentence_count": len(sent_tokenize(main_text))
        },
        "target_document": {
            "word_count": target_word_count,
            "sentence_count": len(sent_tokenize(target_text))
        },
        "similarity": {
            "overall_score": similarity,
            "interpretation": interpret_similarity(similarity)
        },
        "contradictions": {
            "count": len(contradictions),
            "severity": interpret_contradiction_severity(contradictions)
        },
        "exact_matches": {
            "count": len(exact_matches),
            "percentage": calculate_match_percentage(exact_matches, main_text, target_text)
        }
    }

    return summary

def interpret_similarity(score):
    """Interpret the similarity score"""
    if score > 0.9:
        return "Very high similarity - documents are nearly identical"
    elif score > 0.7:
        return "High similarity - documents cover very similar content"
    elif score > 0.5:
        return "Moderate similarity - documents share significant content"
    elif score > 0.3:
        return "Low similarity - documents have some overlap"
    else:
        return "Very low similarity - documents are mostly different"

def interpret_contradiction_severity(contradictions):
    """Interpret the severity of contradictions"""
    if not contradictions:
        return "No contradictions detected"

    avg_score = sum(c["contradiction_score"] for c in contradictions) / len(contradictions)

    if len(contradictions) > 5 and avg_score > 0.8:
        return "High severity - many strong contradictions"
    elif len(contradictions) > 3 or avg_score > 0.8:
        return "Moderate severity - some significant contradictions"
    else:
        return "Low severity - few or weak contradictions"

def calculate_match_percentage(exact_matches, main_text, target_text):
    """Calculate the percentage of exact matches"""
    if not exact_matches:
        return 0.0

    # Sum up the lengths of matched segments
    matched_chars = sum(len(match["main_segment"]) for match in exact_matches)
    total_chars = max(1, len(main_text))  # Avoid division by zero

    return round((matched_chars / total_chars) * 100, 1)

# API endpoints
@app.post("/api/compare_documents/", response_model=SimilarityResponse)
async def compare_documents(main_document: UploadFile = File(...), target_document: UploadFile = File(...)):
    """Compare two documents for similarity and contradictions"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Load models if not already loaded
        load_models()

        # Save files
        main_path = os.path.join(temp_dir, main_document.filename)
        target_path = os.path.join(temp_dir, target_document.filename)

        with open(main_path, "wb") as f:
            shutil.copyfileobj(main_document.file, f)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_document.file, f)

        # Process files
        logger.info("Extracting text from documents...")
        main_text = extract_text(main_path)
        target_text = extract_text(target_path)

        if not main_text or not target_text:
            raise HTTPException(400, "Could not extract text from one or both documents")

        main_processed = preprocess_text(main_text)
        target_processed = preprocess_text(target_text)

        # Segment the texts
        logger.info("Segmenting documents...")
        main_segments = segment_text(main_processed)
        target_segments = segment_text(target_processed)

        # Calculate similarity
        logger.info("Calculating similarity...")
        tfidf_similarity = calculate_tfidf_similarity(main_processed, target_processed)
        semantic_similarity = calculate_semantic_similarity(main_processed, target_processed)

        # Overall similarity is a weighted combination
        overall_similarity = 0.6 * semantic_similarity + 0.4 * tfidf_similarity

        # Find similar segments
        logger.info("Finding similar segments...")
        segment_similarities = get_segment_similarity(main_segments, target_segments)

        # Find exact matches
        logger.info("Finding exact matches...")
        exact_matches = find_exact_matches(main_segments, target_segments)

        # Detect contradictions
        logger.info("Detecting contradictions...")
        contradictions = detect_contradictions(main_segments, target_segments)

        # Generate analysis summary
        logger.info("Generating analysis summary...")
        analysis_summary = generate_analysis_summary(
            main_text, target_text, overall_similarity, contradictions, exact_matches
        )

        logger.info("Analysis complete!")

        return SimilarityResponse(
            overall_similarity=overall_similarity,
            contradictions=contradictions,
            similarity_scores=segment_similarities,
            main_document_text=main_text,
            target_document_text=target_text,
            main_segments=main_segments,
            target_segments=target_segments,
            semantic_similarity=semantic_similarity,
            exact_matches=exact_matches,
            analysis_summary=analysis_summary
        )
    except Exception as e:
        logger.error(f"Error in document comparison: {str(e)}")
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(temp_dir)

@app.get("/api/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": bool(nlp and sentence_model and nli_model)}



FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Document Comparison</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js"></script>
    <style>
        .highlight { background-color: rgba(255, 205, 86, 0.3); }
        .highlight-strong { background-color: rgba(255, 205, 86, 0.5); }
        .contradiction { background-color: rgba(255, 99, 132, 0.3); }
        .contradiction-strong { background-color: rgba(255, 99, 132, 0.5); }
        .match { background-color: rgba(75, 192, 192, 0.3); }
        .match-strong { background-color: rgba(75, 192, 192, 0.5); }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px; height: 36px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .document-view {
            height: 500px; overflow-y: auto;
            white-space: pre-wrap; font-family: system-ui, -apple-system, sans-serif;
            font-size: 14px; line-height: 1.5; padding: 1rem;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .tab-active {
            border-bottom: 2px solid #4f46e5;
            color: #4f46e5;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="text-center mb-8">
            <h1 class="text-3xl font-bold text-indigo-800">Advanced Document Comparison</h1>
            <p class="text-gray-600">Upload two documents to analyze similarity and detect contradictions</p>
        </header>

        <div class="bg-white rounded-lg shadow p-6 mb-8">
            <div class="grid md:grid-cols-2 gap-6">
                <div>
                    <label class="block text-gray-700 mb-2">Main Document</label>
                    <div class="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:bg-gray-50" id="main-drop">
                        <input type="file" id="main-file" class="hidden" accept=".pdf,.docx,.txt">
                        <div class="py-6">
                            <svg class="w-10 h-10 text-gray-400 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                            <p class="mt-2 text-sm text-gray-500">Click to upload</p>
                            <p class="text-xs text-gray-500">PDF, DOCX, or TXT</p>
                        </div>
                        <div id="main-info" class="hidden text-sm font-medium text-gray-600"></div>
                    </div>
                </div>
                <div>
                    <label class="block text-gray-700 mb-2">Target Document</label>
                    <div class="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:bg-gray-50" id="target-drop">
                        <input type="file" id="target-file" class="hidden" accept=".pdf,.docx,.txt">
                        <div class="py-6">
                            <svg class="w-10 h-10 text-gray-400 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                            <p class="mt-2 text-sm text-gray-500">Click to upload</p>
                            <p class="text-xs text-gray-500">PDF, DOCX, or TXT</p>
                        </div>
                        <div id="target-info" class="hidden text-sm font-medium text-gray-600"></div>
                    </div>
                </div>
            </div>
            <div class="mt-6 text-center">
                <button id="analyze-btn" class="px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50" disabled>
                    <span id="btn-text">Analyze Documents</span>
                    <span id="spinner" class="spinner ml-2 hidden"></span>
                </button>
            </div>
        </div>

        <div id="results" class="hidden">
            <div class="bg-white rounded-lg shadow overflow-hidden mb-8">
                <div class="p-6 border-b">
                    <h2 class="text-xl font-bold text-gray-800">Document Analysis Summary</h2>
                    <p class="text-gray-600 text-sm mt-1">Comprehensive comparison results</p>
                </div>

                <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6 p-6">
                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h3 class="font-medium mb-2 text-gray-700">Overall Similarity</h3>
                        <div class="flex items-center mb-2">
                            <div class="w-full bg-gray-200 rounded-full h-4">
                                <div id="similarity-bar" class="bg-indigo-600 h-4 rounded-full" style="width:0%"></div>
                            </div>
                            <span id="similarity-value" class="ml-3 font-medium">0%</span>
                        </div>
                        <p id="similarity-interpretation" class="text-sm text-gray-600">Pending analysis</p>
                    </div>

                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h3 class="font-medium mb-2 text-gray-700">Semantic Similarity</h3>
                        <div class="flex items-center mb-2">
                            <div class="w-full bg-gray-200 rounded-full h-4">
                                <div id="semantic-bar" class="bg-blue-500 h-4 rounded-full" style="width:0%"></div>
                            </div>
                            <span id="semantic-value" class="ml-3 font-medium">0%</span>
                        </div>
                        <p class="text-sm text-gray-600">Based on meaning analysis</p>
                    </div>

                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h3 class="font-medium mb-2 text-gray-700">Contradictions Found</h3>
                        <div class="flex items-center">
                            <span id="contradiction-count" class="text-2xl font-bold text-red-500">0</span>
                            <span class="ml-2">potential issues</span>
                        </div>
                        <p id="contradiction-severity" class="text-sm text-gray-600 mt-1">No contradictions</p>
                    </div>

                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h3 class="font-medium mb-2 text-gray-700">Exact Matches</h3>
                        <div class="flex items-center">
                            <span id="exact-count" class="text-2xl font-bold text-green-500">0</span>
                            <span class="ml-2">matches</span>
                        </div>
                        <p id="exact-percentage" class="text-sm text-gray-600 mt-1">0% of content</p>
                    </div>

                    <div class="bg-gray-50 p-4 rounded-lg col-span-1 md:col-span-2 lg:col-span-1">
                        <h3 class="font-medium mb-2 text-gray-700">Document Statistics</h3>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div>
                                <p class="text-gray-600">Main Document:</p>
                                <p id="main-stats" class="font-medium">- words, - sentences</p>
                            </div>
                            <div>
                                <p class="text-gray-600">Target Document:</p>
                                <p id="target-stats" class="font-medium">- words, - sentences</p>
                            </div>
                        </div>
                    </div>

                    <div class="bg-gray-50 p-4 rounded-lg lg:col-span-3">
                        <h3 class="font-medium mb-3 text-gray-700">Similarity Distribution</h3>
                        <canvas id="similarityChart" height="100"></canvas>
                    </div>
                </div>
            </div>

            <!-- Tabs Navigation -->
            <div class="bg-white rounded-t-lg shadow-sm mb-1">
                <div class="flex border-b">
                    <button id="tab-contradictions" class="tab-btn flex-1 py-3 px-4 text-center hover:bg-gray-50 tab-active">
                        Contradictions
                    </button>
                    <button id="tab-similarities" class="tab-btn flex-1 py-3 px-4 text-center hover:bg-gray-50">
                        Similar Content
                    </button>
                    <button id="tab-exact" class="tab-btn flex-1 py-3 px-4 text-center hover:bg-gray-50">
                        Exact Matches
                    </button>
                    <button id="tab-documents" class="tab-btn flex-1 py-3 px-4 text-center hover:bg-gray-50">
                        Full Documents
                    </button>
                </div>
            </div>

            <!-- Tab Contents -->
            <div class="tab-content-wrapper">
                <!-- Contradictions Tab -->
                <div id="content-contradictions" class="tab-content bg-white rounded-b-lg shadow mb-8">
                    <div class="p-6">
                        <div id="no-contradictions" class="text-gray-600 italic">No contradictions detected</div>
                        <div id="contradiction-list" class="space-y-4 hidden"></div>
                    </div>
                </div>

                <!-- Similarities Tab -->
                <div id="content-similarities" class="tab-content bg-white rounded-b-lg shadow mb-8 hidden">
                    <div class="p-6">
                        <div id="no-similarities" class="text-gray-600 italic">No similar content detected</div>
                        <div id="similarity-list" class="space-y-4 hidden"></div>
                    </div>
                </div>

                <!-- Exact Matches Tab -->
                <div id="content-exact" class="tab-content bg-white rounded-b-lg shadow mb-8 hidden">
                    <div class="p-6">
                        <div id="no-exact" class="text-gray-600 italic">No exact matches detected</div>
                        <div id="exact-list" class="space-y-4 hidden"></div>
                    </div>
                </div>

                <!-- Documents Tab -->
                <div id="content-documents" class="tab-content bg-white rounded-b-lg shadow mb-8 hidden">
                    <div class="grid md:grid-cols-2 gap-0">
                        <div class="border-r">
                            <div class="p-4 border-b">
                                <h3 class="font-medium">Main Document</h3>
                            </div>
                            <div id="main-content" class="document-view"></div>
                        </div>
                        <div>
                            <div class="p-4 border-b">
                                <h3 class="font-medium">Target Document</h3>
                            </div>
                            <div id="target-content" class="document-view"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="error" class="hidden bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mt-8 rounded">
            <p class="font-bold">Error</p>
            <p id="error-message"></p>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Elements
            const mainFile = document.getElementById('main-file');
            const mainDrop = document.getElementById('main-drop');
            const mainInfo = document.getElementById('main-info');

            const targetFile = document.getElementById('target-file');
            const targetDrop = document.getElementById('target-drop');
            const targetInfo = document.getElementById('target-info');

            const analyzeBtn = document.getElementById('analyze-btn');
            const btnText = document.getElementById('btn-text');
            const spinner = document.getElementById('spinner');

            const results = document.getElementById('results');
            const error = document.getElementById('error');
            const errorMsg = document.getElementById('error-message');

            // Tabs
            const tabButtons = document.querySelectorAll('.tab-btn');
            const tabContents = document.querySelectorAll('.tab-content');

            // Chart object
            let similarityChart = null;

            // Setup file upload
            setupFileUpload(mainFile, mainDrop, mainInfo);
            setupFileUpload(targetFile, targetDrop, targetInfo);

            // Setup tabs
            setupTabs();

            // Analyze button
            analyzeBtn.addEventListener('click', analyzeDocuments);

            function setupFileUpload(input, drop, info) {
                drop.addEventListener('click', () => input.click());

                input.addEventListener('change', () => {
                    if (input.files.length) {
                        info.innerHTML = `
                            <div class="flex items-center">
                                <svg class="w-4 h-4 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                                </svg>
                                ${input.files[0].name}
                            </div>
                        `;
                        info.classList.remove('hidden');
                    }
                    updateAnalyzeButton();
                });
            }

            function setupTabs() {
                tabButtons.forEach(button => {
                    button.addEventListener('click', () => {
                        // Remove active class from all buttons
                        tabButtons.forEach(btn => btn.classList.remove('tab-active'));

                        // Add active class to clicked button
                        button.classList.add('tab-active');

                        // Hide all content
                        tabContents.forEach(content => content.classList.add('hidden'));

                        // Show content for active tab
                        const contentId = button.id.replace('tab-', 'content-');
                        document.getElementById(contentId).classList.remove('hidden');
                    });
                });
            }

            function updateAnalyzeButton() {
                analyzeBtn.disabled = !(mainFile.files.length && targetFile.files.length);
            }

            async function analyzeDocuments() {
                analyzeBtn.disabled = true;
                spinner.classList.remove('hidden');
                btnText.textContent = 'Analyzing...';
                results.classList.add('hidden');
                error.classList.add('hidden');

                const formData = new FormData();
                formData.append('main_document', mainFile.files[0]);
                formData.append('target_document', targetFile.files[0]);

                try {
                    const response = await fetch('/api/compare_documents/', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.detail || 'Analysis failed');
                    }

                    const data = await response.json();
                    displayResults(data);
                } catch (e) {
                    errorMsg.textContent = e.message;
                    error.classList.remove('hidden');
                } finally {
                    analyzeBtn.disabled = false;
                    spinner.classList.add('hidden');
                    btnText.textContent = 'Analyze Documents';
                }
            }

            function displayResults(data) {
                results.classList.remove('hidden');

                // Update similarity scores
                const similarity = Math.round(data.overall_similarity * 100);
                const semanticSimilarity = Math.round(data.semantic_similarity * 100);

                document.getElementById('similarity-value').textContent = `${similarity}%`;
                document.getElementById('similarity-bar').style.width = `${similarity}%`;
                document.getElementById('similarity-interpretation').textContent = data.analysis_summary.similarity.interpretation;

                document.getElementById('semantic-value').textContent = `${semanticSimilarity}%`;
                document.getElementById('semantic-bar').style.width = `${semanticSimilarity}%`;

                // Update contradiction info
                const contradictions = data.contradictions;
                document.getElementById('contradiction-count').textContent = contradictions.length;
                document.getElementById('contradiction-severity').textContent = data.analysis_summary.contradictions.severity;

                // Update exact matches
                const exactMatches = data.exact_matches;
                document.getElementById('exact-count').textContent = exactMatches.length;
                document.getElementById('exact-percentage').textContent = `${data.analysis_summary.exact_matches.percentage}% of content`;

                // Update document stats
                document.getElementById('main-stats').textContent =
                    `${data.analysis_summary.main_document.word_count} words, ${data.analysis_summary.main_document.sentence_count} sentences`;
                document.getElementById('target-stats').textContent =
                    `${data.analysis_summary.target_document.word_count} words, ${data.analysis_summary.target_document.sentence_count} sentences`;

                // Display contradictions
                displayContradictions(contradictions);

                // Display similarities
                displaySimilarities(data.similarity_scores);

                // Display exact matches
                displayExactMatches(exactMatches);

                // Display documents
                document.getElementById('main-content').textContent = data.main_document_text;
                document.getElementById('target-content').textContent = data.target_document_text;

                // Create or update similarity distribution chart
                createSimilarityChart(data.similarity_scores);

                // Highlight the first tab by default
                document.getElementById('tab-contradictions').click();
            }

            function displayContradictions(contradictions) {
                const contradictionList = document.getElementById('contradiction-list');
                const noContradictions = document.getElementById('no-contradictions');

                if (contradictions.length) {
                    noContradictions.classList.add('hidden');
                    contradictionList.classList.remove('hidden');
                    contradictionList.innerHTML = '';

                    contradictions.forEach((cont, i) => {
                        const item = document.createElement('div');
                        item.className = 'bg-red-50 border border-red-200 rounded p-4';

                        // Format common concepts
                        let conceptsHtml = '';
                        if (cont.common_concepts && cont.common_concepts.length > 0) {
                            conceptsHtml = `
                                <div class="mt-2 text-sm">
                                    <p class="text-gray-600 mb-1">Common concepts:</p>
                                    <div class="flex flex-wrap gap-1">
                                        ${cont.common_concepts.map(concept =>
                                            `<span class="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs">${concept}</span>`
                                        ).join('')}
                                    </div>
                                </div>
                            `;
                        }

                        item.innerHTML = `
                            <div class="flex justify-between mb-2">
                                <h4 class="font-medium text-red-700">Contradiction #${i+1}</h4>
                                <span class="text-sm bg-red-200 text-red-800 px-2 py-1 rounded-full">
                                    Score: ${Math.round(cont.contradiction_score * 100)}%
                                </span>
                            </div>
                            <div class="grid md:grid-cols-2 gap-4">
                                <div>
                                    <p class="text-xs text-gray-500 mb-1">Main:</p>
                                    <p class="text-sm bg-white p-2 rounded border">${cont.main_segment}</p>
                                </div>
                                <div>
                                    <p class="text-xs text-gray-500 mb-1">Target:</p>
                                    <p class="text-sm bg-white p-2 rounded border">${cont.target_segment}</p>
                                </div>
                            </div>
                            ${conceptsHtml}
                        `;
                        contradictionList.appendChild(item);
                    });
                } else {
                    noContradictions.classList.remove('hidden');
                    contradictionList.classList.add('hidden');
                }
            }

            function displaySimilarities(similarities) {
                const similarityList = document.getElementById('similarity-list');
                const noSimilarities = document.getElementById('no-similarities');

                if (similarities.length) {
                    noSimilarities.classList.add('hidden');
                    similarityList.classList.remove('hidden');
                    similarityList.innerHTML = '';

                    // Sort by score, highest first
                    const sortedSimilarities = [...similarities].sort((a, b) => b.score - a.score);

                    // Display top similarities
                    sortedSimilarities.slice(0, 10).forEach((sim, i) => {
                        const item = document.createElement('div');
                        item.className = 'bg-blue-50 border border-blue-200 rounded p-4';

                        // Format metrics if available
                        let metricsHtml = '';
                        if (sim.metrics) {
                            metricsHtml = `
                                <div class="mt-2 grid grid-cols-4 gap-2 text-xs">
                                    <div class="text-center">
                                        <div class="text-gray-600">TF-IDF</div>
                                        <div class="font-medium">${Math.round(sim.metrics.tfidf * 100)}%</div>
                                    </div>
                                    <div class="text-center">
                                        <div class="text-gray-600">Semantic</div>
                                        <div class="font-medium">${Math.round(sim.metrics.semantic * 100)}%</div>
                                    </div>
                                    <div class="text-center">
                                        <div class="text-gray-600">Fuzzy</div>
                                        <div class="font-medium">${Math.round(sim.metrics.fuzzy * 100)}%</div>
                                    </div>
                                    <div class="text-center">
                                        <div class="text-gray-600">Jaccard</div>
                                        <div class="font-medium">${Math.round(sim.metrics.jaccard * 100)}%</div>
                                    </div>
                                </div>
                            `;
                        }

                        item.innerHTML = `
                            <div class="flex justify-between mb-2">
                                <h4 class="font-medium text-blue-700">Similar Content #${i+1}</h4>
                                <span class="text-sm bg-blue-200 text-blue-800 px-2 py-1 rounded-full">
                                    Overall: ${Math.round(sim.score * 100)}%
                                </span>
                            </div>
                            <div class="grid md:grid-cols-2 gap-4">
                                <div>
                                    <p class="text-xs text-gray-500 mb-1">Main:</p>
                                    <p class="text-sm bg-white p-2 rounded border">${sim.main_segment}</p>
                                </div>
                                <div>
                                    <p class="text-xs text-gray-500 mb-1">Target:</p>
                                    <p class="text-sm bg-white p-2 rounded border">${sim.target_segment}</p>
                                </div>
                            </div>
                            ${metricsHtml}
                        `;
                        similarityList.appendChild(item);
                    });
                } else {
                    noSimilarities.classList.remove('hidden');
                    similarityList.classList.add('hidden');
                }
            }

            function displayExactMatches(matches) {
                const exactList = document.getElementById('exact-list');
                const noExact = document.getElementById('no-exact');

                if (matches.length) {
                    noExact.classList.add('hidden');
                    exactList.classList.remove('hidden');
                    exactList.innerHTML = '';

                    // Sort by similarity score
                    const sortedMatches = [...matches].sort((a, b) => b.similarity - a.similarity);

                    sortedMatches.forEach((match, i) => {
                        const item = document.createElement('div');
                        item.className = 'bg-green-50 border border-green-200 rounded p-4';
                        item.innerHTML = `
                            <div class="flex justify-between mb-2">
                                <h4 class="font-medium text-green-700">Exact Match #${i+1}</h4>
                                <span class="text-sm bg-green-200 text-green-800 px-2 py-1 rounded-full">
                                    Match: ${Math.round(match.similarity * 100)}%
                                </span>
                            </div>
                            <div class="grid md:grid-cols-2 gap-4">
                                <div>
                                    <p class="text-xs text-gray-500 mb-1">Main:</p>
                                    <p class="text-sm bg-white p-2 rounded border">${match.main_segment}</p>
                                </div>
                                <div>
                                    <p class="text-xs text-gray-500 mb-1">Target:</p>
                                    <p class="text-sm bg-white p-2 rounded border">${match.target_segment}</p>
                                </div>
                            </div>
                        `;
                        exactList.appendChild(item);
                    });
                } else {
                    noExact.classList.remove('hidden');
                    exactList.classList.add('hidden');
                }
            }

            function createSimilarityChart(similarityScores) {
                // Destroy previous chart if it exists
                if (similarityChart) {
                    similarityChart.destroy();
                }

                // If no similarity scores, don't create the chart
                if (!similarityScores || similarityScores.length === 0) {
                    return;
                }

                // Group similarity scores into ranges
                const ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
                const labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'];
                const counts = new Array(ranges.length - 1).fill(0);

                similarityScores.forEach(item => {
                    for (let i = 0; i < ranges.length - 1; i++) {
                        if (item.score >= ranges[i] && item.score < ranges[i + 1]) {
                            counts[i]++;
                            break;
                        }
                    }
                });

                // Create the chart
                const ctx = document.getElementById('similarityChart').getContext('2d');
                similarityChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Number of Similar Segments',
                            data: counts,
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.5)',
                                'rgba(255, 159, 64, 0.5)',
                                'rgba(255, 205, 86, 0.5)',
                                'rgba(75, 192, 192, 0.5)',
                                'rgba(54, 162, 235, 0.5)'
                            ],
                            borderColor: [
                                'rgb(255, 99, 132)',
                                'rgb(255, 159, 64)',
                                'rgb(255, 205, 86)',
                                'rgb(75, 192, 192)',
                                'rgb(54, 162, 235)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `${context.parsed.y} segments (${(context.parsed.y / similarityScores.length * 100).toFixed(1)}%)`;
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Number of segments'
                                },
                                ticks: {
                                    precision: 0
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Similarity score range'
                                }
                            }
                        }
                    }
                });
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)

# Start the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)