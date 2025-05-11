document.addEventListener('DOMContentLoaded', () => {
    // Initialize PDF.js
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

    const mainDocument = document.getElementById('mainDocument');
    const targetDocument = document.getElementById('targetDocument');
    const mainFileName = document.getElementById('mainFileName');
    const targetFileName = document.getElementById('targetFileName');
    const compareBtn = document.getElementById('compareBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const similarityValue = document.getElementById('similarityValue');
    const riskScoreValue = document.getElementById('riskScoreValue');
    const riskScoreGauge = document.getElementById('riskScoreGauge');
    const findingsList = document.getElementById('findingsList');
    const clauseAnalysisList = document.getElementById('clauseAnalysisList');
    const missingClausesList = document.getElementById('missingClausesList');
    const contradictionsList = document.getElementById('contradictionsList');
    const segmentsList = document.getElementById('segmentsList');
    const mainDownload = document.getElementById('mainDownload');
    const targetDownload = document.getElementById('targetDownload');
    const analysisDownload = document.getElementById('analysisDownload');
    const mainPdfViewer = document.getElementById('mainPdfViewer');
    const targetPdfViewer = document.getElementById('targetPdfViewer');

    // Store PDF data for rendering
    let mainPdfData = null;
    let targetPdfData = null;
    let mainPdfDocument = null;
    let targetPdfDocument = null;

    // Update file names when files are selected
    mainDocument.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            mainFileName.textContent = file.name;
            mainPdfData = await file.arrayBuffer();
            checkFilesSelected();
        }
    });

    targetDocument.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            targetFileName.textContent = file.name;
            targetPdfData = await file.arrayBuffer();
            checkFilesSelected();
        }
    });

    // Enable compare button only when both files are selected
    function checkFilesSelected() {
        compareBtn.disabled = !(mainDocument.files[0] && targetDocument.files[0]);
    }

    // Handle file comparison
    compareBtn.addEventListener('click', async () => {
        const formData = new FormData();
        formData.append('main_document', mainDocument.files[0]);
        formData.append('target_document', targetDocument.files[0]);

        // Show loading state
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        compareBtn.disabled = true;

        try {
            const response = await fetch('/compare-documents', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to compare documents');
            }

            const data = await response.json();
            await displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while comparing documents. Please try again.');
        } finally {
            loading.classList.add('hidden');
            compareBtn.disabled = false;
        }
    });

    // Render PDF with highlights
    async function renderPdf(pdfData, container, highlights = []) {
        const pdf = await pdfjsLib.getDocument({ data: pdfData }).promise;
        const numPages = pdf.numPages;
        
        for (let pageNum = 1; pageNum <= numPages; pageNum++) {
            const page = await pdf.getPage(pageNum);
            // Increase scale for better readability
            const viewport = page.getViewport({ scale: 2.0 });
            
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;
            
            const renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            
            await page.render(renderContext).promise;
            
            // Add highlights if any
            if (highlights.length > 0) {
                const pageHighlights = highlights.filter(h => h.page === pageNum);
                if (pageHighlights.length > 0) {
                    const highlightLayer = document.createElement('div');
                    highlightLayer.className = 'highlight-layer';
                    highlightLayer.style.position = 'absolute';
                    highlightLayer.style.top = '0';
                    highlightLayer.style.left = '0';
                    highlightLayer.style.width = '100%';
                    highlightLayer.style.height = '100%';
                    highlightLayer.style.pointerEvents = 'none'; // Allow clicking through highlights
                    
                    pageHighlights.forEach(highlight => {
                        const highlightElement = document.createElement('div');
                        highlightElement.className = 'highlight-contradiction';
                        highlightElement.style.position = 'absolute';
                        // Scale the coordinates to match the viewport
                        highlightElement.style.left = `${highlight.bbox[0] * 2}px`;
                        highlightElement.style.top = `${highlight.bbox[1] * 2}px`;
                        highlightElement.style.width = `${(highlight.bbox[2] - highlight.bbox[0]) * 2}px`;
                        highlightElement.style.height = `${(highlight.bbox[3] - highlight.bbox[1]) * 2}px`;
                        highlightLayer.appendChild(highlightElement);
                    });
                    
                    const wrapper = document.createElement('div');
                    wrapper.style.position = 'relative';
                    wrapper.appendChild(canvas);
                    wrapper.appendChild(highlightLayer);
                    container.appendChild(wrapper);
                } else {
                    container.appendChild(canvas);
                }
            } else {
                container.appendChild(canvas);
            }
        }
    }

    // Display results
    async function displayResults(data) {
        // Clear previous PDFs
        mainPdfViewer.innerHTML = '';
        targetPdfViewer.innerHTML = '';

        // Instead of using mainPdfData, fetch the highlighted PDF
        const mainHighlightedUrl = '/' + data.highlighted_documents.main;
        const targetHighlightedUrl = '/' + data.highlighted_documents.target;

        const mainHighlightedResponse = await fetch(mainHighlightedUrl);
        const mainHighlightedBuffer = await mainHighlightedResponse.arrayBuffer();

        const targetHighlightedResponse = await fetch(targetHighlightedUrl);
        const targetHighlightedBuffer = await targetHighlightedResponse.arrayBuffer();

        // Now render these with PDF.js
        await renderPdf(mainHighlightedBuffer, mainPdfViewer);
        await renderPdf(targetHighlightedBuffer, targetPdfViewer);

        // Update risk score
        riskScoreValue.textContent = `${data.analysis.overall_risk_score.toFixed(1)}%`;
        riskScoreGauge.style.width = `${data.analysis.overall_risk_score}%`;

        // Update similarity
        similarityValue.textContent = `${data.analysis.similarity.toFixed(1)}%`;

        // Update key findings
        findingsList.innerHTML = '';
        data.analysis.key_findings.forEach(finding => {
            const li = document.createElement('li');
            li.textContent = finding;
            findingsList.appendChild(li);
        });

        // Update clause analysis
        clauseAnalysisList.innerHTML = '';
        data.analysis.clause_analysis.forEach(clause => {
            const clauseElement = document.createElement('div');
            clauseElement.className = 'clause-item';
            
            const header = document.createElement('div');
            header.className = 'clause-header';
            
            const title = document.createElement('h5');
            title.textContent = 'Clause Analysis';
            
            const riskBadge = document.createElement('span');
            riskBadge.className = `risk-badge ${clause.risk_level.toLowerCase()}`;
            riskBadge.textContent = clause.risk_level;
            
            header.appendChild(title);
            header.appendChild(riskBadge);
            
            const content = document.createElement('div');
            content.className = 'clause-content';
            
            const text = document.createElement('p');
            text.textContent = clause.clause_text;
            
            const implications = document.createElement('div');
            implications.className = 'implications';
            implications.innerHTML = `
                <h6>Legal Implications:</h6>
                <ul>
                    ${clause.legal_implications.map(imp => `<li>${imp}</li>`).join('')}
                </ul>
            `;
            
            const recommendations = document.createElement('div');
            recommendations.className = 'recommendations';
            recommendations.innerHTML = `
                <h6>Recommendations:</h6>
                <ul>
                    ${clause.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            `;
            
            const standards = document.createElement('div');
            standards.className = 'standards';
            standards.innerHTML = `
                <h6>Industry Standards:</h6>
                <ul>
                    ${clause.industry_standards.map(std => `<li>${std}</li>`).join('')}
                </ul>
            `;
            
            content.appendChild(text);
            content.appendChild(implications);
            content.appendChild(recommendations);
            content.appendChild(standards);
            
            clauseElement.appendChild(header);
            clauseElement.appendChild(content);
            clauseAnalysisList.appendChild(clauseElement);
        });

        // Update missing clauses
        missingClausesList.innerHTML = '';
        data.analysis.missing_clauses.forEach(clause => {
            const li = document.createElement('li');
            li.textContent = clause;
            missingClausesList.appendChild(li);
        });

        // Update contradictions
        contradictionsList.innerHTML = '';
        data.analysis.contradictions.forEach(contradiction => {
            const li = document.createElement('li');
            li.textContent = contradiction;
            contradictionsList.appendChild(li);
        });

        // Update segments
        segmentsList.innerHTML = '';
        data.analysis.contradictory_segments.forEach(segment => {
            const li = document.createElement('li');
            li.textContent = segment;
            segmentsList.appendChild(li);
        });

        // Show results
        results.classList.remove('hidden');
    }
}); 