document.addEventListener('DOMContentLoaded', () => {
    const mainDocument = document.getElementById('mainDocument');
    const targetDocument = document.getElementById('targetDocument');
    const mainFileName = document.getElementById('mainFileName');
    const targetFileName = document.getElementById('targetFileName');
    const compareBtn = document.getElementById('compareBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const similarityValue = document.getElementById('similarityValue');
    const contradictionsList = document.getElementById('contradictionsList');
    const segmentsList = document.getElementById('segmentsList');
    const mainDownload = document.getElementById('mainDownload');
    const targetDownload = document.getElementById('targetDownload');

    // Update file names when files are selected
    mainDocument.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            mainFileName.textContent = file.name;
            checkFilesSelected();
        }
    });

    targetDocument.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            targetFileName.textContent = file.name;
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
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while comparing documents. Please try again.');
        } finally {
            loading.classList.add('hidden');
            compareBtn.disabled = false;
        }
    });

    // Display results
    function displayResults(data) {
        // Update similarity
        similarityValue.textContent = `${data.analysis.similarity.toFixed(1)}%`;

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

        // Update download links
        mainDownload.href = `/download/${data.highlighted_documents.main.split('/').pop()}`;
        targetDownload.href = `/download/${data.highlighted_documents.target.split('/').pop()}`;

        // Show results
        results.classList.remove('hidden');
    }
}); 