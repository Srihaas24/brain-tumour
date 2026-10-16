/**
 * NeuroScan AI — Client-Side Application Logic
 * Brain Tumor Detection & Multi-Class Categorization Dashboard
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('file-input');
    const emptyState = document.getElementById('empty-state');
    const resultsContainer = document.getElementById('results-container');
    const scanningOverlay = document.getElementById('scanning-overlay');
    
    // Viewer Elements
    const originalScanImg = document.getElementById('original-scan-img');
    const heatmapScanImg = document.getElementById('heatmap-scan-img');
    const opacitySlider = document.getElementById('opacity-slider');
    const opacityVal = document.getElementById('opacity-val');

    // Diagnosis Output Elements
    const diagBanner = document.getElementById('diagnosis-banner');
    const diagCategory = document.getElementById('diag-category');
    const diagStatus = document.getElementById('diag-status');
    const confidenceVal = document.getElementById('confidence-val');
    
    // Probability Bars
    const barGlioma = document.getElementById('bar-glioma');
    const valGlioma = document.getElementById('val-glioma');
    const barMeningioma = document.getElementById('bar-meningioma');
    const valMeningioma = document.getElementById('val-meningioma');
    const barPituitary = document.getElementById('bar-pituitary');
    const valPituitary = document.getElementById('val-pituitary');
    const barNotumor = document.getElementById('bar-notumor');
    const valNotumor = document.getElementById('val-notumor');

    // Clinical Details
    const clinicSeverity = document.getElementById('clinic-severity');
    const clinicOrigin = document.getElementById('clinic-origin');
    const clinicDesc = document.getElementById('clinic-desc');
    const clinicSymptoms = document.getElementById('clinic-symptoms');
    const clinicActions = document.getElementById('clinic-actions');

    // Sample cards
    const sampleCards = document.querySelectorAll('.sample-card');

    // Modals
    const trainModal = document.getElementById('train-modal');
    const statsModal = document.getElementById('stats-modal');
    const openTrainBtn = document.getElementById('btn-open-train');
    const openStatsBtn = document.getElementById('btn-open-stats');
    const closeBtns = document.querySelectorAll('.modal-close, .modal-overlay');

    // --- Drag and Drop Events ---
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // --- Sample Card Selection ---
    sampleCards.forEach(card => {
        card.addEventListener('click', () => {
            const sampleName = card.getAttribute('data-sample');
            handleSampleSelect(sampleName);
        });
    });

    // --- Opacity Slider for Heatmap Cross-fade ---
    if (opacitySlider) {
        opacitySlider.addEventListener('input', (e) => {
            const val = e.target.value;
            opacityVal.textContent = `${val}%`;
            if (heatmapScanImg) {
                heatmapScanImg.style.opacity = val / 100;
            }
        });
    }

    // --- API Handlers ---
    async function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid MRI image (JPEG, PNG).');
            return;
        }

        const formData = new FormData();
        formData.append('scan', file);
        await runAnalysis(formData);
    }

    async function handleSampleSelect(sampleName) {
        const formData = new FormData();
        formData.append('sample_name', sampleName);
        await runAnalysis(formData);
    }

    async function runAnalysis(formData) {
        // Show scan loading state
        emptyState.style.display = 'none';
        resultsContainer.style.display = 'block';
        scanningOverlay.classList.add('scanning');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to analyze scan');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Analysis error:', error);
            alert(`Analysis Error: ${error.message}`);
        } finally {
            scanningOverlay.classList.remove('scanning');
        }
    }

    function displayResults(data) {
        // 1. Populate Images
        originalScanImg.src = data.original_image_b64;
        heatmapScanImg.src = data.overlay_image_b64;
        heatmapScanImg.style.opacity = (opacitySlider.value / 100);

        // 2. Populate Primary Diagnosis Banner
        diagBanner.className = 'diagnosis-banner ' + (data.has_tumor ? 'has-tumor' : 'no-tumor');
        diagCategory.textContent = data.has_tumor ? 'Intracranial Neoplasm Detected' : 'Normal Brain Anatomy';
        diagStatus.textContent = data.display_name;
        diagStatus.style.color = data.badge_color;
        confidenceVal.textContent = `${data.confidence}%`;

        // 3. Populate Probability Distribution Bars
        const probs = data.probabilities;
        updateBar(barGlioma, valGlioma, probs.glioma || 0);
        updateBar(barMeningioma, valMeningioma, probs.meningioma || 0);
        updateBar(barPituitary, valPituitary, probs.pituitary || 0);
        updateBar(barNotumor, valNotumor, probs.no_tumor || 0);

        // 4. Clinical Guidance
        clinicSeverity.textContent = data.severity;
        clinicSeverity.style.color = data.badge_color;
        clinicOrigin.textContent = data.origin;
        clinicDesc.textContent = data.description;
        clinicSymptoms.textContent = data.symptoms;
        clinicActions.textContent = data.clinical_actions;

        // Smooth scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function updateBar(barElement, valElement, percentage) {
        valElement.textContent = `${percentage}%`;
        barElement.style.width = `${percentage}%`;
    }

    // --- Modal Controls ---
    if (openTrainBtn) {
        openTrainBtn.addEventListener('click', (e) => {
            e.preventDefault();
            trainModal.classList.add('active');
        });
    }

    if (openStatsBtn) {
        openStatsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            statsModal.classList.add('active');
        });
    }

    closeBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            if (e.target === btn || btn.classList.contains('modal-close')) {
                trainModal.classList.remove('active');
                statsModal.classList.remove('active');
            }
        });
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            trainModal.classList.remove('active');
            statsModal.classList.remove('active');
        }
    });

    // Auto-load first sample on start so user immediately sees rich interface
    setTimeout(() => {
        handleSampleSelect('sample_glioma.jpg');
    }, 400);
});
