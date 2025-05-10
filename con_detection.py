import nltk
# Download all required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab', quiet=True)  # This might not exist, but we'll try quietly
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import shutil
import tempfile
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import PyPDF2
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nest_asyncio

# Initialize
nest_asyncio.apply()

# Create app
app = FastAPI(title="Document Matching and Contradiction Detection")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NLI model initialization
MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = None
model = None

def load_nli_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

# Response model
class SimilarityResponse(BaseModel):
    overall_similarity: float
    contradictions: List[Dict[str, Any]]
    similarity_scores: List[Dict[str, Any]]
    main_document_text: str
    target_document_text: str

# Text processing functions
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    text = ""
    doc = docx.Document(file_path)
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as file:
            return file.read()

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
    return re.sub(r'\s+', ' ', text).strip()

def segment_text(text):
    return [s.strip() for s in sent_tokenize(text) if s.strip()]

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

def get_segment_similarity(main_segs, target_segs):
    if not main_segs or not target_segs:
        return []

    vectorizer = TfidfVectorizer()
    all_segs = main_segs + target_segs
    tfidf = vectorizer.fit_transform(all_segs)

    results = []
    for i, main_seg in enumerate(main_segs):
        main_vec = tfidf[i:i+1]
        best = {"score": 0.0, "main_segment": main_seg, "target_segment": "", "main_index": i, "target_index": -1}

        for j, target_seg in enumerate(target_segs):
            target_vec = tfidf[len(main_segs) + j:len(main_segs) + j + 1]
            score = float(cosine_similarity(main_vec, target_vec)[0][0])
            if score > best["score"]:
                best = {"score": score, "main_segment": main_seg, "target_segment": target_seg, "main_index": i, "target_index": j}

        if best["score"] > 0.5:
            results.append(best)

    return results

def detect_contradictions(main_segs, target_segs):
    load_nli_model()
    contradictions = []

    for i, main_seg in enumerate(main_segs):
        for j, target_seg in enumerate(target_segs):
            if len(main_seg.split()) < 5 or len(target_seg.split()) < 5:
                continue

            try:
                inputs = tokenizer(main_seg, target_seg, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).squeeze(0)

                # Get the probability for contradiction (typically index 2 for NLI)
                contradiction_score = float(probs[2].item()) if len(probs) > 2 else 0.0

                if contradiction_score > 0.7:
                    contradictions.append({
                        "main_segment": main_seg,
                        "target_segment": target_seg,
                        "contradiction_score": contradiction_score,
                        "main_index": i,
                        "target_index": j
                    })
            except Exception as e:
                print(f"Error processing contradiction: {str(e)}")
                continue

    return sorted(contradictions, key=lambda x: x["contradiction_score"], reverse=True)

# API endpoints
@app.post("/api/compare_documents/", response_model=SimilarityResponse)
async def compare_documents(main_document: UploadFile = File(...), target_document: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        # Save files
        main_path = os.path.join(temp_dir, main_document.filename)
        target_path = os.path.join(temp_dir, target_document.filename)

        with open(main_path, "wb") as f:
            shutil.copyfileobj(main_document.file, f)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_document.file, f)

        # Process files
        main_text = extract_text(main_path)
        target_text = extract_text(target_path)
        main_processed = preprocess_text(main_text)
        target_processed = preprocess_text(target_text)

        # Analyze
        similarity = calculate_similarity(main_processed, target_processed)
        main_segments = segment_text(main_processed)
        target_segments = segment_text(target_processed)
        segment_scores = get_segment_similarity(main_segments, target_segments)
        contradictions = detect_contradictions(main_segments, target_segments)

        return SimilarityResponse(
            overall_similarity=similarity,
            contradictions=contradictions,
            similarity_scores=segment_scores,
            main_document_text=main_text,
            target_document_text=target_text
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(temp_dir)

@app.get("/api/health/")
async def health_check():
    return {"status": "healthy"}

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Matching & Contradiction Detection</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <style>
        .highlight { background-color: rgba(255, 205, 86, 0.3); }
        .contradiction { background-color: rgba(255, 99, 132, 0.3); }
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
            white-space: pre-wrap; font-family: monospace;
            font-size: 14px; line-height: 1.5; padding: 1rem;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="text-center mb-8">
            <h1 class="text-3xl font-bold text-gray-800">Document Comparison</h1>
            <p class="text-gray-600">Upload two documents to analyze similarity and contradictions</p>
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
                <button id="analyze-btn" class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50" disabled>
                    <span id="btn-text">Analyze Documents</span>
                    <span id="spinner" class="spinner ml-2 hidden"></span>
                </button>
            </div>
        </div>

        <div id="results" class="hidden">
            <div class="bg-white rounded-lg shadow p-6 mb-8">
                <h2 class="text-xl font-bold mb-4">Results</h2>
                <div class="grid md:grid-cols-2 gap-6 mb-6">
                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h3 class="font-medium mb-2">Similarity Score</h3>
                        <div class="flex items-center">
                            <div class="w-full bg-gray-200 rounded-full h-4">
                                <div id="similarity-bar" class="bg-blue-600 h-4 rounded-full" style="width:0%"></div>
                            </div>
                            <span id="similarity-value" class="ml-3 font-medium">0%</span>
                        </div>
                    </div>
                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h3 class="font-medium mb-2">Contradictions Found</h3>
                        <div class="flex items-center">
                            <span id="contradiction-count" class="text-2xl font-bold text-red-500">0</span>
                            <span class="ml-2">potential issues</span>
                        </div>
                    </div>
                </div>

                <div id="contradictions" class="mb-6">
                    <h3 class="font-medium mb-3">Contradiction Details</h3>
                    <div id="no-contradictions" class="text-gray-600 italic">No contradictions detected</div>
                    <div id="contradiction-list" class="space-y-4 hidden"></div>
                </div>
            </div>

            <div class="grid md:grid-cols-2 gap-6">
                <div class="bg-white rounded-lg shadow">
                    <div class="p-4 border-b">
                        <h3 class="font-medium">Main Document</h3>
                    </div>
                    <div id="main-content" class="document-view"></div>
                </div>
                <div class="bg-white rounded-lg shadow">
                    <div class="p-4 border-b">
                        <h3 class="font-medium">Target Document</h3>
                    </div>
                    <div id="target-content" class="document-view"></div>
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

            // Setup file upload
            setupFileUpload(mainFile, mainDrop, mainInfo);
            setupFileUpload(targetFile, targetDrop, targetInfo);

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

                // Update similarity
                const similarity = Math.round(data.overall_similarity * 100);
                document.getElementById('similarity-value').textContent = `${similarity}%`;
                document.getElementById('similarity-bar').style.width = `${similarity}%`;

                // Update contradictions
                const contradictions = data.contradictions;
                document.getElementById('contradiction-count').textContent = contradictions.length;

                const contradictionList = document.getElementById('contradiction-list');
                const noContradictions = document.getElementById('no-contradictions');

                if (contradictions.length) {
                    noContradictions.classList.add('hidden');
                    contradictionList.classList.remove('hidden');
                    contradictionList.innerHTML = '';

                    contradictions.forEach((cont, i) => {
                        const item = document.createElement('div');
                        item.className = 'bg-red-50 border border-red-200 rounded p-4';
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
                        `;
                        contradictionList.appendChild(item);
                    });
                } else {
                    noContradictions.classList.remove('hidden');
                    contradictionList.classList.add('hidden');
                }

                // Display documents
                document.getElementById('main-content').textContent = data.main_document_text;
                document.getElementById('target-content').textContent = data.target_document_text;
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