/**
 * script.js -- Interactive logic for Healthcare Patient Risk Mining Dashboard
 */

// Presets data for instant demonstration
const PRESETS = {
  low: {
    age: 38,
    sex: 0,
    cp: 2,
    trestbps: 115,
    chol: 190,
    fbs: 0,
    restecg: 0,
    thalach: 178,
    exang: 0,
    oldpeak: 0.0,
    slope: 1,
    ca: 0,
    thal: 3
  },
  moderate: {
    age: 55,
    sex: 1,
    cp: 3,
    trestbps: 135,
    chol: 245,
    fbs: 0,
    restecg: 1,
    thalach: 148,
    exang: 0,
    oldpeak: 1.2,
    slope: 2,
    ca: 0,
    thal: 3
  },
  high: {
    age: 64,
    sex: 1,
    cp: 4,
    trestbps: 160,
    chol: 286,
    fbs: 1,
    restecg: 2,
    thalach: 108,
    exang: 1,
    oldpeak: 2.8,
    slope: 2,
    ca: 3,
    thal: 7
  }
};

let currentBatchResults = [];

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupPredictionForm();
  setupCSVUpload();
  checkAPIHealth();
  loadEDAGallery();
  loadModelEvaluation();
});

/* -------------------------------------------------------------------------
   TABS MANAGEMENT
   ------------------------------------------------------------------------- */
function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });
}

/* -------------------------------------------------------------------------
   HEALTH STATUS
   ------------------------------------------------------------------------- */
async function checkAPIHealth() {
  const dot = document.getElementById('system-status-dot');
  const text = document.getElementById('system-status-text');
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      if (data.model_loaded) {
        dot.style.backgroundColor = 'var(--risk-low)';
        text.textContent = 'Models Online & Ready';
      } else {
        dot.style.backgroundColor = 'var(--risk-mod)';
        text.textContent = 'Model Initializing';
      }
    }
  } catch (e) {
    dot.style.backgroundColor = 'var(--risk-high)';
    text.textContent = 'API Offline';
  }
}

/* -------------------------------------------------------------------------
   SINGLE PATIENT PREDICTION
   ------------------------------------------------------------------------- */
function loadPreset(type) {
  const data = PRESETS[type];
  if (!data) return;

  for (const [key, val] of Object.entries(data)) {
    const el = document.getElementById(`input-${key}`);
    if (el) {
      el.value = val;
    }
  }

  // Trigger prediction automatically for smooth demo
  submitPrediction();
}

function setupPredictionForm() {
  const form = document.getElementById('prediction-form');
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitPrediction();
  });
}

async function submitPrediction() {
  const form = document.getElementById('prediction-form');
  const formData = new FormData(form);
  const payload = {};

  formData.forEach((value, key) => {
    payload[key] = parseFloat(value);
  });

  const btn = document.getElementById('btn-submit-predict');
  const originalText = btn.innerHTML;
  btn.innerHTML = `<span class="status-dot"></span> Analyzing Patient Vitals...`;
  btn.disabled = true;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await res.json();

    if (!res.ok) {
      alert(`Prediction failed: ${result.error || 'Server error'}`);
      return;
    }

    renderPredictionResult(result);

  } catch (err) {
    alert(`Error connecting to prediction API: ${err.message}`);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

function renderPredictionResult(res) {
  const prob = (res.probability * 100).toFixed(1);
  const probNoDis = (res.probability_no_disease * 100).toFixed(1);
  const category = res.risk ? res.risk.category : 'Unknown';

  // Elements
  const badge = document.getElementById('res-risk-badge');
  const label = document.getElementById('res-risk-label');
  const probVal = document.getElementById('res-prob-val');
  const gaugeFill = document.getElementById('res-gauge-fill');
  const classVal = document.getElementById('res-classification');
  const probNoDisVal = document.getElementById('res-prob-nodisease');
  const descVal = document.getElementById('res-description');

  probVal.innerHTML = `${prob}<span>%</span>`;
  gaugeFill.style.width = `${Math.min(Math.max(prob, 5), 100)}%`;
  classVal.textContent = res.prediction_label;
  probNoDisVal.textContent = `${probNoDis}%`;
  descVal.textContent = res.risk ? res.risk.description : '';

  badge.className = 'risk-badge';
  if (category === 'Low Risk') {
    badge.classList.add('low');
    label.textContent = '🟢 Low Risk';
  } else if (category === 'Moderate Risk') {
    badge.classList.add('moderate');
    label.textContent = '🟡 Moderate Risk';
  } else {
    badge.classList.add('high');
    label.textContent = '🔴 High Risk';
  }
}

/* -------------------------------------------------------------------------
   BATCH CSV PREDICTION
   ------------------------------------------------------------------------- */
function setupCSVUpload() {
  const dropzone = document.getElementById('csv-dropzone');
  const fileInput = document.getElementById('csv-file-input');

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleCSVFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleCSVFile(e.target.files[0]);
    }
  });
}

async function handleCSVFile(file) {
  if (!file.name.endsWith('.csv')) {
    alert('Please upload a valid .csv file.');
    return;
  }

  const title = document.getElementById('upload-status-title');
  title.textContent = `Processing "${file.name}"...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/predict-csv', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      alert(`Batch processing error: ${data.error || 'Server error'}`);
      title.textContent = 'Upload failed. Try again.';
      return;
    }

    title.textContent = `Successfully processed ${data.total_records} patients!`;
    renderBatchResults(data);

  } catch (err) {
    alert(`Upload failed: ${err.message}`);
    title.textContent = 'Drag & Drop your CSV file here, or click to browse';
  }
}

function renderBatchResults(data) {
  currentBatchResults = data.records;
  const container = document.getElementById('batch-results-container');
  container.style.display = 'block';

  // Stats
  document.getElementById('batch-stat-total').textContent = data.total_records;
  document.getElementById('batch-stat-high').textContent = data.risk_distribution['High Risk'] || 0;
  document.getElementById('batch-stat-mod').textContent = data.risk_distribution['Moderate Risk'] || 0;
  document.getElementById('batch-stat-low').textContent = data.risk_distribution['Low Risk'] || 0;

  // Table
  const tbody = document.getElementById('batch-table-body');
  tbody.innerHTML = '';

  data.records.forEach((row, i) => {
    const tr = document.createElement('tr');
    const risk = row.risk_category || 'Unknown';
    let riskBadgeClass = 'low';
    if (risk === 'Moderate Risk') riskBadgeClass = 'moderate';
    if (risk === 'High Risk') riskBadgeClass = 'high';

    const pVal = row.probability !== null ? `${(row.probability * 100).toFixed(1)}%` : 'N/A';

    tr.innerHTML = `
      <td>#${i + 1}</td>
      <td>${row.age || '-'} / ${row.sex === 1 ? 'M' : 'F'}</td>
      <td>Type ${row.cp || '-'}</td>
      <td>${row.trestbps || '-'} mmHg</td>
      <td>${row.chol || '-'} mg/dL</td>
      <td>${row.thalach || '-'} bpm</td>
      <td>${row.oldpeak !== undefined ? row.oldpeak : '-'}</td>
      <td><span class="badge-tag" style="font-size: 0.72rem; color: var(--risk-${riskBadgeClass}); border-color: var(--risk-${riskBadgeClass});">${risk}</span></td>
      <td><strong>${pVal}</strong></td>
    `;
    tbody.appendChild(tr);
  });

  // Scroll to results
  container.scrollIntoView({ behavior: 'smooth' });
}

function downloadSampleCSV() {
  const headers = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n";
  const rows = [
    "52,1,1,125,212,0,1,168,0,1.0,2,2,3",
    "53,1,1,140,203,1,0,155,1,3.1,0,0,7",
    "70,1,1,145,174,0,1,125,1,2.6,0,0,7",
    "61,1,1,148,203,0,1,161,0,0.0,2,1,3",
    "62,0,1,138,294,1,1,106,0,1.9,1,3,2",
    "58,1,1,112,230,0,0,165,0,2.5,1,1,7",
    "58,1,1,114,318,0,2,140,0,4.4,0,3,1",
    "44,1,2,120,263,0,1,173,0,0.0,2,0,3",
    "60,1,1,140,185,0,0,155,0,3.0,1,0,3",
    "67,1,1,120,229,0,0,129,1,2.6,1,2,7"
  ].join("\n");

  const blob = new Blob([headers + rows], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "heart_disease_sample_cohort.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function exportBatchResults() {
  if (!currentBatchResults || currentBatchResults.length === 0) {
    alert('No batch records to export.');
    return;
  }

  const keys = Object.keys(currentBatchResults[0]);
  const headerLine = keys.join(',') + '\n';
  const rowLines = currentBatchResults.map(r => keys.map(k => `"${r[k] !== undefined ? r[k] : ''}"`).join(',')).join('\n');

  const blob = new Blob([headerLine + rowLines], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "screened_patient_risk_results.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/* -------------------------------------------------------------------------
   EDA GALLERY & DESCRIPTIONS
   ------------------------------------------------------------------------- */
const PLOT_DESCRIPTIONS = {
  'correlation_heatmap.png': {
    title: 'Clinical Correlation Matrix',
    desc: 'Bivariate correlations across all 13 clinical attributes showing strong relationships between thalach, oldpeak, and target.'
  },
  'clustering_analysis.png': {
    title: 'Unsupervised K-Means & PCA',
    desc: 'Patient clustering in reduced 2D PCA feature space discovering natural groupings in clinical risk profiles.'
  },
  'risk_factor_associations.png': {
    title: 'Risk Factor Associations',
    desc: 'Categorical cross-tabulations demonstrating disease prevalence across chest pain types and fluoroscopy major vessels.'
  },
  'clinical_distributions.png': {
    title: 'Clinical Parameter Distributions',
    desc: 'Histograms with KDE curves of serum cholesterol, resting blood pressure, and maximum exercise heart rate.'
  },
  'age_distribution.png': {
    title: 'Age Density by Diagnosis',
    desc: 'Age demographics comparison between healthy cohorts and patients diagnosed with coronary heart disease.'
  },
  'feature_comparisons.png': {
    title: 'Feature Distributions Across Classes',
    desc: 'Box plots comparing key numeric indicators (oldpeak, thalach, chol, trestbps) split by disease outcome.'
  },
  'target_distribution.png': {
    title: 'Target Outcome Balance',
    desc: 'Proportion of disease absence (0) vs presence (1) verifying clean balance in the Cleveland cohort.'
  },
  'grouped_summary.png': {
    title: 'Demographic Risk Segmentation',
    desc: 'Stratification of disease rate grouped by age quartiles and biological sex.'
  }
};

const EVAL_DESCRIPTIONS = {
  'roc_curves.png': {
    title: 'Multi-Model ROC Curves',
    desc: 'Receiver Operating Characteristic comparison; Logistic Regression achieved 89.3% mean cross-validated AUC.'
  },
  'pr_curves.png': {
    title: 'Precision-Recall Curves',
    desc: 'Precision vs Recall trade-offs essential for evaluating healthcare screening thresholds and minimizing false negatives.'
  },
  'confusion_matrices.png': {
    title: 'Confusion Matrices Matrix',
    desc: 'True Positives, False Positives, True Negatives, and False Negatives evaluated on hold-out test patients.'
  },
  'feature_importance.png': {
    title: 'Feature Importance & Weights',
    desc: 'Global feature importance rankings indicating primary diagnostic drivers (ca, thal, thalach, oldpeak).'
  },
  'model_comparison.png': {
    title: 'Performance Benchmark Comparison',
    desc: 'Cross-validated Accuracy, Precision, Recall, and F1 across all four supervised learning algorithms.'
  }
};

async function loadEDAGallery() {
  const container = document.getElementById('eda-gallery-grid');
  try {
    const res = await fetch('/api/eda');
    const data = await res.json();
    const plots = data.plots || [];

    // Filter EDA specific plots
    const edaPlots = plots.filter(p => !['roc_curves.png', 'pr_curves.png', 'confusion_matrices.png', 'feature_importance.png', 'model_comparison.png'].includes(p));

    container.innerHTML = '';
    edaPlots.forEach(filename => {
      const info = PLOT_DESCRIPTIONS[filename] || {
        title: filename.replace('.png', '').replace(/_/g, ' ').toUpperCase(),
        desc: 'Exploratory data analysis plot mined from patient clinical records.'
      };

      const card = document.createElement('div');
      card.className = 'gallery-card';
      card.onclick = () => openModal(`/static/plots/${filename}`);
      card.innerHTML = `
        <div class="gallery-img-container">
          <img src="/static/plots/${filename}" alt="${info.title}" loading="lazy">
        </div>
        <div class="gallery-card-body">
          <div class="gallery-card-title">${info.title}</div>
          <div class="gallery-card-desc">${info.desc}</div>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load EDA plots:', err);
  }
}

async function loadModelEvaluation() {
  const container = document.getElementById('eval-gallery-grid');
  try {
    const res = await fetch('/api/eda');
    const data = await res.json();
    const plots = data.plots || [];

    const evalPlots = plots.filter(p => ['roc_curves.png', 'pr_curves.png', 'confusion_matrices.png', 'feature_importance.png', 'model_comparison.png'].includes(p));

    container.innerHTML = '';
    evalPlots.forEach(filename => {
      const info = EVAL_DESCRIPTIONS[filename] || {
        title: filename.replace('.png', '').replace(/_/g, ' ').toUpperCase(),
        desc: 'Model evaluation metric visualization.'
      };

      const card = document.createElement('div');
      card.className = 'gallery-card';
      card.onclick = () => openModal(`/static/plots/${filename}`);
      card.innerHTML = `
        <div class="gallery-img-container">
          <img src="/static/plots/${filename}" alt="${info.title}" loading="lazy">
        </div>
        <div class="gallery-card-body">
          <div class="gallery-card-title">${info.title}</div>
          <div class="gallery-card-desc">${info.desc}</div>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load Eval plots:', err);
  }
}

/* -------------------------------------------------------------------------
   LIGHTBOX MODAL
   ------------------------------------------------------------------------- */
function openModal(imgSrc) {
  const modal = document.getElementById('lightbox-modal');
  const img = document.getElementById('modal-img');
  img.src = imgSrc;
  modal.classList.add('active');
}

function closeModal() {
  const modal = document.getElementById('lightbox-modal');
  modal.classList.remove('active');
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});
