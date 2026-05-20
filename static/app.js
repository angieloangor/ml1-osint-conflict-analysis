// Initialize Lucide icons
lucide.createIcons();

// --- Sidebar and Panels Logic ---
const sidebar = document.getElementById('sidebar');
const leftPanel = document.getElementById('left-panel');
const bottomPanel = document.getElementById('bottom-panel');
const topPanel = document.getElementById('top-panel');
const body = document.body;

function closeAllPanels() {
    sidebar.classList.add('hidden-right');
    leftPanel.classList.add('hidden-left');
    bottomPanel.classList.add('hidden-bottom');
    topPanel.classList.add('hidden-top');
    body.classList.remove('panel-right-open', 'panel-left-open', 'panel-bottom-open', 'panel-top-open');
}

document.getElementById('sidebar-toggle').addEventListener('click', () => {
    const isClosed = sidebar.classList.contains('hidden-right');
    closeAllPanels();
    if (isClosed) { sidebar.classList.remove('hidden-right'); body.classList.add('panel-right-open'); }
});

document.getElementById('left-panel-toggle').addEventListener('click', () => {
    const isClosed = leftPanel.classList.contains('hidden-left');
    closeAllPanels();
    if (isClosed) { leftPanel.classList.remove('hidden-left'); body.classList.add('panel-left-open'); }
});

document.getElementById('bottom-panel-toggle').addEventListener('click', () => {
    const isClosed = bottomPanel.classList.contains('hidden-bottom');
    closeAllPanels();
    if (isClosed) { bottomPanel.classList.remove('hidden-bottom'); body.classList.add('panel-bottom-open'); }
});

document.getElementById('top-panel-toggle').addEventListener('click', () => {
    const isClosed = topPanel.classList.contains('hidden-top');
    closeAllPanels();
    if (isClosed) { topPanel.classList.remove('hidden-top'); body.classList.add('panel-top-open'); }
});

document.getElementById('close-bottom-panel').addEventListener('click', closeAllPanels);
document.getElementById('close-top-panel').addEventListener('click', closeAllPanels);

// --- Tabs Logic (Right Panel) ---
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active-tab'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active-tab');
    });
});

// --- Modal Logic (Models) ---
const modal = document.getElementById('model-modal');
const modalTitle = document.getElementById('modal-title');
const modalDesc = document.getElementById('modal-desc');
const modalMetrics = document.getElementById('modal-metrics');
const metricAcc = document.getElementById('metric-acc');
const metricPrec = document.getElementById('metric-prec');
const metricF1 = document.getElementById('metric-f1');
const modalImage = document.getElementById('modal-image');

const modelData = {
    "classic": {
        title: "NLP Clásico",
        desc: "Línea base interpretable. Utiliza Bag of Words y TF-IDF para evaluar la polaridad inicial y frecuencia de términos. Es fundamental para asegurar que los modelos más complejos realmente aporten valor sobre métodos más simples.",
        hasMetrics: false, image: "/outputs/figures/confusion_matrix_nb.png"
    },
    "modern": {
        title: "NLP Moderno",
        desc: "Mapea las noticias a un espacio denso (embeddings) utilizando sentence-transformers. Permite entender los 'vecindarios semánticos' de las narrativas sin depender de coincidencias exactas de palabras clave.",
        hasMetrics: false, image: "/outputs/advanced_figures/semantic_space_umap.png"
    },
    "bertopic": {
        title: "BERTopic",
        desc: "Algoritmo de modelado temático que descubre los clústeres latentes de discusión. Es vital para este objetivo OSINT porque revela automáticamente narrativas como 'Crisis Humanitaria' sin necesidad de pre-etiquetar los datos.",
        hasMetrics: false, image: "/outputs/figures/topic_distribution.png"
    },
    "gradientboosting": {
        title: "Gradient Boosting",
        desc: "El clasificador más potente del ensamble. Aporta precisión quirúrgica en la clasificación de alertas utilizando las etiquetas débiles construidas durante la extracción OSINT.",
        hasMetrics: true, acc: "88.8%", prec: "75.5%", f1: "72.5%", image: "/outputs/advanced_figures/confusion_matrix_gradientboosting.png"
    },
    "randomforest": {
        title: "Random Forest",
        desc: "Modelo de ensamble robusto basado en múltiples árboles de decisión. Proporciona estabilidad frente al ruido en los datos extraídos de noticias crudas (RSS/GDELT).",
        hasMetrics: true, acc: "85.1%", prec: "71.3%", f1: "66.4%", image: "/outputs/advanced_figures/confusion_matrix_randomforest.png"
    },
    "faiss": {
        title: "Búsqueda Semántica FAISS",
        desc: "Índice de similitud semántica. Permite a los analistas buscar conceptos abstractos ('tensiones marítimas') y encontrar reportes relevantes aunque no contengan esas palabras exactas.",
        hasMetrics: false, image: "/outputs/figures/semantic_embedding_scatter.png"
    }
};

document.querySelectorAll('.model-card').forEach(card => {
    card.addEventListener('click', () => {
        const id = card.dataset.model;
        const data = modelData[id];
        if(!data) return;

        modalTitle.textContent = data.title;
        modalDesc.textContent = data.desc;
        modalImage.src = data.image;

        if (data.hasMetrics) {
            modalMetrics.classList.remove('hidden');
            metricAcc.textContent = data.acc;
            metricPrec.textContent = data.prec;
            metricF1.textContent = data.f1;
        } else {
            modalMetrics.classList.add('hidden');
        }
        
        modal.classList.remove('hidden-modal');
    });
});

document.getElementById('close-modal').addEventListener('click', () => {
    modal.classList.add('hidden-modal');
});

// --- API Logic ---
const placeholderText = document.getElementById('placeholder-text');
const predictionResult = document.getElementById('prediction-result');
const countryNameEl = document.getElementById('country-name');
const riskLevelText = document.getElementById('risk-level-text');
const riskProgress = document.getElementById('risk-progress');
const predictionDesc = document.getElementById('prediction-desc');
const themesList = document.getElementById('themes-list');
const historyText = document.getElementById('history-text');
const newsList = document.getElementById('news-list');

async function fetchPrediction(lat, lng) {
    // Open right sidebar if closed
    if (sidebar.classList.contains('hidden-right')) {
        sidebar.classList.remove('hidden-right');
        body.classList.add('panel-right-open');
    }

    // Show loading state
    placeholderText.classList.add('hidden');
    predictionResult.classList.remove('hidden');
    countryNameEl.textContent = "Analyzing...";
    predictionDesc.textContent = "Querying OSINT ML model...";
    riskProgress.style.width = "0%";
    themesList.innerHTML = '';

    try {
        const response = await fetch(`/api/predict?lat=${lat}&lng=${lng}`);
        const data = await response.json();
        
        // Update UI
        countryNameEl.textContent = data.name;
        riskLevelText.textContent = data.risk_level;
        
        // Reset classes
        riskLevelText.className = '';
        riskProgress.className = 'progress-fill';
        
        if (data.risk_score > 70) {
            riskLevelText.classList.add('risk-high');
            riskProgress.style.backgroundColor = 'var(--accent-red)';
        } else if (data.risk_score > 40) {
            riskLevelText.classList.add('risk-medium');
            riskProgress.style.backgroundColor = '#f59e0b'; // amber
        } else {
            riskLevelText.classList.add('risk-low');
            riskProgress.style.backgroundColor = '#10b981'; // emerald
        }
        
        // Animate progress bar
        setTimeout(() => {
            riskProgress.style.width = `${data.risk_score}%`;
        }, 100);

        predictionDesc.textContent = data.prediction;

        // Add badges
        data.top_themes.forEach(theme => {
            const span = document.createElement('span');
            span.className = 'theme-badge';
            span.textContent = theme;
            themesList.appendChild(span);
        });

        // Add History & War Context
        if (data.history) {
            historyText.textContent = data.history;
        } else {
            historyText.textContent = "No hay datos históricos disponibles para esta región.";
        }
        
        const warContextText = document.getElementById('war-context-text');
        if (data.war_context) {
            warContextText.textContent = data.war_context;
        } else {
            warContextText.textContent = "No hay información de guerra activa registrada.";
        }

        // Add News
        newsList.innerHTML = '';
        if (data.news && data.news.length > 0) {
            data.news.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="news-date">${item.date}</span>
                    <span class="news-headline">${item.headline}</span>
                    <span class="news-source">${item.source}</span>
                `;
                newsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.innerHTML = `<span class="news-source">Sin alertas OSINT recientes.</span>`;
            newsList.appendChild(li);
        }

    } catch (error) {
        console.error("Error fetching prediction:", error);
        countryNameEl.textContent = "Error";
        predictionDesc.textContent = "Could not reach the intelligence backend.";
    }
}


// --- Globe Initialization ---
// Data for the fixed red dots (Iran, Israel, USA)
const keyLocations = [
    { lat: 31.5, lng: 34.75, name: 'Israel', size: 1.5, color: '#ef4444' }, // Israel
    { lat: 32.42, lng: 53.68, name: 'Iran', size: 2.0, color: '#ef4444' }, // Iran
    { lat: 38.0, lng: -97.0, name: 'USA', size: 2.5, color: '#ef4444' }    // USA
];

const globe = Globe()
    (document.getElementById('globeViz'))
    .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
    .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
    
    // Add point markers for key locations
    .pointsData(keyLocations)
    .pointLat('lat')
    .pointLng('lng')
    .pointColor('color')
    .pointAltitude(0.05)
    .pointRadius('size')
    .pointsMerge(false)
    
    // Glow effect for the points
    .pointResolution(32)
    
    // Interaction
    .onGlobeClick(({ lat, lng }) => {
        // Move camera to look at the clicked point slightly
        globe.pointOfView({ lat: lat, lng: lng, altitude: 2 }, 1000);
        
        // Fetch and show data in sidebar
        fetchPrediction(lat, lng);
    });

// Make it spin slowly
globe.controls().autoRotate = true;
globe.controls().autoRotateSpeed = 0.5;

// Focus initially on the Middle East
globe.pointOfView({ lat: 32, lng: 44, altitude: 2.5 });

// Handle window resize
window.addEventListener('resize', () => {
    globe.width(window.innerWidth);
    globe.height(window.innerHeight);
});

// Add a slight red pulse animation to the point markers via CSS is hard in canvas,
// so we use pointAltitude dynamically for a breathing effect
let altitude = 0.05;
let growing = true;
setInterval(() => {
    altitude = growing ? altitude + 0.002 : altitude - 0.002;
    if (altitude >= 0.1) growing = false;
    if (altitude <= 0.05) growing = true;
    globe.pointAltitude(altitude);
}, 50);
