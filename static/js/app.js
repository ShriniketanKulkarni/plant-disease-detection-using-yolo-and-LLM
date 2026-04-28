document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('image-upload');
    const dropArea = document.getElementById('drop-area');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const predictBtn = document.getElementById('predict-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const resultSection = document.getElementById('result-section');
    const resultImage = document.getElementById('result-image');
    const newScanBtn = document.getElementById('new-scan-btn');
    
    let currentFile = null;

    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    uploadInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            const reader = new FileReader();
            reader.readAsDataURL(currentFile);
            reader.onloadend = function() {
                imagePreview.src = reader.result;
                dropArea.classList.add('hidden');
                previewContainer.classList.remove('hidden');
            }
        }
    }

    resetBtn.addEventListener('click', () => {
        currentFile = null;
        uploadInput.value = '';
        previewContainer.classList.add('hidden');
        dropArea.classList.remove('hidden');
    });

    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI state changes
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
        predictBtn.dataset.originalText = predictBtn.innerText;
        predictBtn.innerText = 'Analyzing...';
        document.getElementById('action-buttons').classList.add('hidden');
        document.getElementById('scanner-overlay').classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                displayResults(data);
            } else {
                alert(data.error || 'Server error occurred');
                resetUI();
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server.');
            resetUI();
        }
    });

    function displayResults(data) {
        document.getElementById('scanner-overlay').classList.add('hidden');
        document.getElementById('upload-section').classList.add('hidden');
        resultSection.classList.remove('hidden');

        resultImage.src = 'data:image/jpeg;base64,' + data.image;
        
        if (data.gemini_disease || data.gemini_health_report) {
            document.getElementById('main-disease-name').innerText = data.gemini_disease || "Analysis Complete";
            document.getElementById('health-report-content').innerHTML = marked.parse(data.gemini_health_report || "No data");
            document.getElementById('symptoms-content').innerHTML = marked.parse(data.gemini_symptoms || "No symptoms listed");
            document.getElementById('treatment-content').innerHTML = marked.parse(data.gemini_treatment || "No treatment suggestions");
        } else {
             let diseaseName = "Healthy Plant 🌿";
             if (data.detections && data.detections.length > 0) {
                 diseaseName = data.detections[0].class_name.replace(/__/g, ' ').replace(/_/g, ' ');
             }
             document.getElementById('main-disease-name').innerText = diseaseName;
             document.getElementById('health-report-content').innerHTML = "AI verification failed or no key provided.";
             document.getElementById('symptoms-content').innerHTML = "-";
             document.getElementById('treatment-content').innerHTML = "-";
        }
    }

    function resetUI() {
        predictBtn.classList.remove('loading');
        predictBtn.disabled = false;
        predictBtn.innerText = predictBtn.dataset.originalText || 'Scan for Disease';
        document.getElementById('scanner-overlay').classList.add('hidden');
        document.getElementById('action-buttons').classList.remove('hidden');
        resultSection.classList.add('hidden');
        document.getElementById('upload-section').classList.remove('hidden');
        resetBtn.click();
    }

    newScanBtn.addEventListener('click', resetUI);
});
