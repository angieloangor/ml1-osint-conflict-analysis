from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="OSINT Intelligence API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data based on the OSINT model description
DISPLACEMENT_RISK_DATA = {
    "israel": {
        "name": "Israel & Palestinian Territories",
        "risk_level": "High",
        "risk_score": 89,
        "prediction": "The NLP model has detected a high volume of keywords related to 'military', 'escalation', and 'humanitarian' near this region. Our pipeline predicts an immediate high risk of population displacement due to localized conflict events (ACLED) and escalating diplomatic tensions.",
        "top_themes": ["Military Escalation", "Humanitarian Crisis", "Geopolitical Tensions"],
        "history": "Desde la perspectiva histórica reciente monitoreada en este corpus, el área ha sido el epicentro de tensiones continuas. Tras múltiples escaladas de violencia, el discurso OSINT refleja una movilización militar constante, ataques cruzados y una profunda crisis humanitaria que genera desplazamientos masivos internos e internacionales, todo fuertemente cubierto por medios globales y regionales.",
        "war_context": "En el contexto de la guerra activa, Israel se enfrenta no solo a actores locales (Palestina, Gaza), sino al Eje de la Resistencia apoyado por Irán (Hezbolá en Líbano, milicias en Siria/Irak y los Hutíes en Yemen). Esta interconexión genera un conflicto de espectro completo (tierra, aire, misiles balísticos). Las señales OSINT muestran una alta frecuencia de incidentes cinéticos y una dependencia crítica del apoyo militar de EE. UU. (interceptores, logística).",
        "news": [
            {"date": "2024-05-18", "headline": "Nuevas alertas de defensa aérea activadas en la región norte", "source": "GDELT Network"},
            {"date": "2024-05-17", "headline": "Llamados internacionales a cese al fuego en el Consejo de Seguridad", "source": "BBC RSS"},
            {"date": "2024-05-16", "headline": "Reportes (ACLED): Incremento de incidentes cinéticos en áreas fronterizas", "source": "OSINT Monitor"}
        ]
    },
    "iran": {
        "name": "Iran",
        "risk_level": "Medium-High",
        "risk_score": 75,
        "prediction": "Signals from GDELT and news RSS feeds indicate significant diplomatic activity and sanctions discussions. While immediate physical displacement risk is localized, there is a moderate to high economic displacement and migration risk based on sentiment analysis.",
        "top_themes": ["Sanctions", "Diplomacy", "Energy"],
        "history": "En el contexto de este corpus, Irán representa un actor geopolítico crucial. La narrativa OSINT destaca su papel en el 'Eje de la Resistencia', enfrentando fuertes regímenes de sanciones occidentales. Su historia reciente en el análisis muestra una diplomacia de confrontación, desarrollo de capacidades balísticas y nucleares, y ciberataques, lo que genera tensiones indirectas en múltiples frentes de Medio Oriente sin llegar aún a una invasión territorial directa a gran escala.",
        "war_context": "La estrategia de Irán es fundamentalmente asimétrica. Utiliza proxies regionales para hostigar a Israel y a bases de EE. UU., evitando, en la medida de lo posible, una guerra abierta en su propio territorio. Los datos OSINT y satelitales (UKMTO, Sentinel) muestran disrupciones en el estrecho de Ormuz y el Mar Rojo por parte de sus aliados. La tensión nuclear y balística mantiene a las defensas regionales en máxima alerta constante.",
        "news": [
            {"date": "2024-05-18", "headline": "Impacto económico: Análisis de nuevas sanciones al sector energético", "source": "Al Jazeera RSS"},
            {"date": "2024-05-15", "headline": "Movimientos navales detectados cerca del Estrecho de Ormuz (UKMTO)", "source": "Sentinel Hub Data"},
            {"date": "2024-05-12", "headline": "Declaraciones oficiales rechazan advertencias de organismos internacionales", "source": "Google News RSS"}
        ]
    },
    "usa": {
        "name": "United States",
        "risk_level": "Low",
        "risk_score": 12,
        "prediction": "The OSINT corpus indicates the USA is primarily a diplomatic and military actor in the conflict narrative, rather than a site of physical displacement. Displacement risk here is structurally low.",
        "top_themes": ["Diplomacy", "Military Support", "Policy"],
        "history": "Estados Unidos es perfilado en el corpus como el principal patrocinador diplomático y militar de Israel, a la vez que actúa como contrapeso estratégico contra Irán en la región. Su historia reciente en esta narrativa OSINT está marcada por intentos de disuasión, despliegue de grupos de portaaviones en el Mediterráneo y el Mar Rojo, y un intenso debate político interno sobre la financiación y el enfoque de la política exterior en Medio Oriente.",
        "war_context": "EE. UU. actúa como disuasor global y escudo balístico para la región. Las señales muestran el envío continuo de armamento y activos navales. Además, lidera coaliciones internacionales contra los ataques en el Mar Rojo. El riesgo principal para EE. UU. según el pipeline OSINT no es el desplazamiento en su territorio, sino la erosión de su capital diplomático y el riesgo de verse arrastrado a un conflicto directo con Irán.",
        "news": [
            {"date": "2024-05-18", "headline": "Paquete de asistencia militar aprobado en la legislatura", "source": "Google News RSS"},
            {"date": "2024-05-16", "headline": "Departamento de Estado anuncia nueva ronda de sanciones financieras", "source": "BBC RSS"},
            {"date": "2024-05-14", "headline": "Protestas civiles detectadas en múltiples campus universitarios", "source": "Bluesky OSINT"}
        ]
    }
}

@app.get("/api/predict")
def predict_displacement(lat: float, lng: float):
    # Simple bounding box approximation for the demo
    
    # Israel / Palestine approx bounding box
    if 29.0 <= lat <= 34.0 and 34.0 <= lng <= 36.0:
        return DISPLACEMENT_RISK_DATA["israel"]
    
    # Iran approx bounding box
    if 25.0 <= lat <= 40.0 and 44.0 <= lng <= 63.0:
        return DISPLACEMENT_RISK_DATA["iran"]
    
    # USA approx bounding box (contiguous)
    if 24.0 <= lat <= 49.0 and -125.0 <= lng <= -66.0:
        return DISPLACEMENT_RISK_DATA["usa"]
        
    return {
        "name": "Other Region",
        "risk_level": "Low/Unknown",
        "risk_score": 5,
        "prediction": "The current ML model is focused on the Iran-Israel-US conflict narrative. For this region, our data pipeline has not detected significant conflict-related OSINT signals.",
        "top_themes": ["General News"],
        "history": "No hay un análisis histórico profundo para esta región en el actual corpus enfocado en Medio Oriente y actores clave.",
        "war_context": "La región seleccionada no presenta señales directas de involucramiento militar o logístico en el eje Irán-Israel analizado por el modelo."
    }

# Mount static directories
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "..", "static")
outputs_dir = os.path.join(base_dir, "..", "outputs")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
