# AI Contract Reviewer

An intelligent contract review system that uses AI to analyze and compare legal documents, identifying contradictions, discrepancies, and potential risks.

## Features

- **Document Comparison**: Compare two legal documents and identify contradictions and discrepancies
- **AI-Powered Analysis**: Uses Google's Gemini AI model for intelligent document analysis
- **PDF Processing**: Supports both text-based and scanned PDFs with OCR capabilities
- **Highlighted Results**: Generates highlighted versions of documents showing contradictions
- **Risk Assessment**: Provides overall risk scores and key findings
- **Missing Clauses**: Identifies important clauses missing in either document

## Setup

1. Create and activate a virtual environment:

For Linux/Mac (Bash):
```bash
python -m venv venv
source venv/bin/activate
```

For Windows (cmd):
```cmd
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root and add:
```
GEMINI_API_KEY=your_api_key_here
```

4. Install additional requirements:

For Windows:
- Tesseract OCR v5.5.1 ([Download](https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.1))
  - Install to default location: `C:\Program Files\Tesseract-OCR`
  - Add to PATH: `C:\Program Files\Tesseract-OCR`
- Poppler v24.08.0 ([Download](https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0))
  - Extract to: `C:\Program Files\poppler-24.08.0`
  - Add to PATH: `C:\Program Files\poppler-24.08.0\Library\bin`

To add to PATH in Windows:
1. Open System Properties (Win + Pause/Break)
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System Variables", find and select "Path"
5. Click "Edit"
6. Click "New" and add the paths mentioned above
7. Click "OK" on all windows

Important: After adding the paths to the system PATH variable, you must restart your terminal/command prompt for the changes to take effect. The application will look for these paths in the following order:
1. System PATH environment variable
2. Default installation locations
3. Project-specific paths

For Linux/Mac:
- Tesseract OCR: Install via package manager
- Poppler: Install via package manager

## Running the Application

To run the application in development mode:
```bash
uvicorn main:app --reload
```

The API will be available at:
- Main API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative API Documentation: http://localhost:8000/redoc

## API Endpoints

### Document Comparison
- `POST /compare`: Compare two PDF documents
  - Input: Two PDF files (main_document and target_document)
  - Output: Analysis results with highlighted contradictions

### PDF Processing
- `POST /convert-pdf`: Convert scanned PDF to text-based PDF
  - Input: PDF file
  - Output: Path to converted PDF

### Utility Endpoints
- `GET /`: Web interface for document comparison
- `GET /health`: Health check endpoint
- `GET /temp/{file_path}`: Download processed files

## Analysis Features

The system analyzes documents for:
- Delivery terms and deadlines
- Functional and non-functional requirements
- Responsibilities and obligations
- Payment terms and penalties
- Timeline and project durations
- Evaluation criteria
- Confidentiality and IP rights
- Dispute resolution procedures
- Jurisdiction and governing law
- Scope creep prevention
- Missing or contradictory clauses

## Response Format

The comparison endpoint returns:
```json
{
    "analysis": {
        "similarity": number,
        "contradictions": [...],
        "contradictory_segments": [...],
        "missing_clauses": [...],
        "overall_risk_score": number,
        "key_findings": [...]
    },
    "highlighted_documents": {
        "main": "path/to/main.pdf",
        "target": "path/to/target.pdf"
    }
}
```