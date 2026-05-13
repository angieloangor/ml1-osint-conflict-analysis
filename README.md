# Pipeline OSINT para ML1

Proyecto listo para recolectar datos de GDELT, BBC RSS, Al Jazeera RSS, Google News RSS, OpenSky y NASA FIRMS, guardar cada fuente por separado y construir `data/dataset_integrado.csv`.

## Estructura Recomendada

```text
Proyecto_final_ml/
  osint_pipeline.py
  requirements.txt
  .env.example
  .env
  data/
    gdelt.csv
    bbc_rss.csv
    aljazeera_rss.csv
    google_news_rss.csv
    opensky.csv
    nasa_firms.csv
    dataset_integrado.csv
  notebooks/
    exploracion.ipynb
```

La carpeta `data/` se crea automaticamente.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y coloca `NASA_FIRMS_MAP_KEY`. OpenSky puede correr anonimo, pero OAuth2 mejora cuota si configuras `OPENSKY_CLIENT_ID` y `OPENSKY_CLIENT_SECRET`.

NASA FIRMS usa exclusivamente la variable `NASA_FIRMS_MAP_KEY`. El endpoint configurado es:

```text
https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/20,10,80,50/7
```

Ese bbox es Medio Oriente ampliado (`west=20`, `south=10`, `east=80`, `north=50`), no mundial. Si la consulta de 7 dias devuelve 0 filas, el pipeline permite una prueba tecnica de 14 dias con el mismo bbox regional mediante `NASA_FIRMS_EMPTY_RETRY_DAY_RANGE=14`. Si la API de FIRMS rechaza rangos superiores al limite vigente del servicio, el pipeline lo registra y prueba el maximo permitido informado por NASA sin cambiar el bbox regional.

## Ejecucion Como Script

```bash
python osint_pipeline.py
```

Solo algunas fuentes:

```bash
python osint_pipeline.py --sources bbc_rss,aljazeera_rss,google_news_rss,opensky
```

Probar GDELT con menos carga:

```bash
python osint_pipeline.py --sources gdelt --gdelt-timespan 6h --gdelt-maxrecords 25 --disable-cache
```

Extraer NASA FIRMS para Medio Oriente:

```bash
python osint_pipeline.py --sources nasa_firms --disable-cache
```

Probar FIRMS con 14 dias sin cambiar el bbox regional:

```bash
python osint_pipeline.py --sources nasa_firms --nasa-day-range 14 --disable-cache
```

## Ejecucion En Jupyter

```python
from osint_pipeline import PipelineConfig, configure_logging, run_pipeline

configure_logging("INFO")
config = PipelineConfig.from_env()
results = run_pipeline(config)

dataset = results["integrated"]
dataset.head()
```

Desactivar una API en Notebook:

```python
config.enabled_sources["gdelt"] = False
results = run_pipeline(config)
```

## Activar O Desactivar APIs

En `.env`:

```text
ENABLED_SOURCES=gdelt,bbc_rss,aljazeera_rss,google_news_rss,opensky,nasa_firms
DISABLED_SOURCES=gdelt
```

Por terminal:

```bash
python osint_pipeline.py --sources gdelt,bbc_rss
```

## Como Agregar Nuevas APIs

1. Crea una funcion `fetch_mi_api(session, config, rate_limiter) -> pd.DataFrame | None`.
2. Devuelve columnas normalizadas: `timestamp`, `source`, `title`, `text`, `url`, `country`, `lat`, `lon`, `value`.
3. Llama `ensure_normalized(df, "mi_api")`.
4. Agrega salida en `SOURCE_OUTPUTS`.
5. Agrega bandera en `DEFAULT_ENABLED_SOURCES`.
6. Registra la funcion en `FETCHERS`.

## Que Hace Cada Bloque Del Codigo

- `DEFAULT_COLUMNS`: define el contrato unico para ML.
- `SOURCE_OUTPUTS`: asigna un CSV por fuente.
- `PipelineConfig`: centraliza credenciales, endpoints, cuotas, cache y parametros.
- `PipelineConfig.from_env()`: lee `.env` y evita hardcodear claves.
- `RateLimiter`: evita llamadas demasiado seguidas a la misma fuente.
- `OpenSkyTokenManager`: usa OAuth2 si hay credenciales y cae a anonimo si falla.
- `build_session()`: reutiliza conexiones HTTP y aplica `User-Agent`.
- `request_with_retries()`: maneja errores de conexion, HTTP 429, HTTP 5xx, `Retry-After`, backoff exponencial y jitter.
- `safe_json()`: valida que las APIs JSON no devuelvan HTML, texto o JSON roto.
- `is_project_relevant()` y `filter_thematic_news()`: filtran fuentes noticiosas para sostener el foco Iran-Israel-EE. UU./Medio Oriente.
- `fetch_gdelt()`: llama GDELT DOC 2.0 con `mode=artlist`, `format=json`, `maxrecords=100`, cache, pausa local, backoff, `Retry-After` y `User-Agent`.
- `fetch_rss_feed()`: parsea BBC y Al Jazeera con `feedparser`.
- `fetch_google_news_rss()`: consulta Google News RSS con queries especificas sobre Iran, Israel, EE. UU. y Medio Oriente, y guarda `google_news_rss.csv`.
- `fetch_opensky()`: transforma state vectors de OpenSky a filas ML como snapshot operacional de movilidad aerea regional.
- `fetch_nasa_firms()`: lee FIRMS CSV usando `NASA_FIRMS_MAP_KEY`, bbox regional `20,10,80,50`, `VIIRS_SNPP_NRT`, intento principal de 7 dias, fallback opcional a 14 dias si sale vacio, ajuste al limite vigente informado por NASA si el API rechaza el rango, retries y validacion de `lat`, `lon`, `timestamp`, `confidence` y `satellite`.
- `integrate_datasets()`: une fuentes exitosas o CSV existentes disponibles, deduplica conservando puntos geograficos distintos y crea dataset integrado aunque otras fallen o devuelvan 0 filas.
- `log_source_summary()`: muestra resumen por fuente y filas finales integradas.
- `main()`: permite ejecutar como script sin usar `sys.exit()`.

## Decisiones para aumentar cobertura sin perder coherencia temática

- Las fuentes siguen siendo publicas, gratuitas y justificables: GDELT, RSS de medios, Google News RSS, OpenSky y NASA FIRMS.
- La region sigue siendo Medio Oriente ampliado. NASA FIRMS usa `NASA_FIRMS_AREA=20,10,80,50`; OpenSky conserva un bbox regional (`OPENSKY_BBOX=24,34,40,64`) para no diluir el contexto.
- No se usa bbox mundial como configuracion final. Un bbox mundial solo podria usarse como prueba tecnica aislada, no para el dataset del informe.
- La unidad de analisis sigue siendo coherente: noticias/eventos OSINT, movilidad aerea operacional regional y detecciones FIRMS dentro del area de estudio.
- GDELT sube de `GDELT_MAXRECORDS=50` a `GDELT_MAXRECORDS=100`, pero mantiene una consulta tematica sobre Iran, Israel, EE. UU., conflicto, escalada, ataques y diplomacia.
- El control de rate limit no cambia: cache local, `User-Agent`, `GDELT_MIN_INTERVAL_SECONDS`, backoff exponencial y respeto de `Retry-After`.
- NASA FIRMS pasa de 1 a 7 dias para aumentar registros sin salir del bbox regional. Si sigue vacio, se puede probar 14 dias con `NASA_FIRMS_EMPTY_RETRY_DAY_RANGE=14`; si NASA informa un limite menor, el pipeline cae al maximo permitido y lo deja en logs.
- Google News RSS agrega tres queries especificas: `Iran Israel conflict`, `Iran Israel escalation` e `Iran US Israel Middle East`.
- Las fuentes noticiosas pasan por un filtro tematico simple antes de integrarse para reducir ruido fuera del eje Iran-Israel-EE. UU.
- El objetivo sigue siendo construir un sistema OSINT multifuente para ML1, no un dataset global sin foco geografico ni tematico.

## Recomendaciones Para Evitar Bloqueos De GDELT

- Usa cache local: `USE_CACHE=true` y `CACHE_TTL_MINUTES=60`.
- Mantén `GDELT_MAXRECORDS=100` para la corrida principal y bajalo a 25 o 50 durante pruebas rapidas.
- Usa `GDELT_TIMESPAN` corto: `6h`, `12h`, `24h`.
- No ejecutes celdas en bucle desde Jupyter.
- Deja `GDELT_MIN_INTERVAL_SECONDS=30` o mas si recibes 429.
- Respeta `Retry-After`; el pipeline ya lo hace.
- `BACKOFF_MAX_SECONDS` limita el backoff calculado; si GDELT envia `Retry-After`, se respeta esa espera.
- Usa un `User-Agent` identificable y academico.
- Si GDELT sigue bloqueado, desactivalo temporalmente y trabaja con RSS/OpenSky/FIRMS.

## EDA Inicial

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/dataset_integrado.csv", parse_dates=["timestamp"])
df.info()
df.head()

df["source"].value_counts().plot(kind="bar", title="Filas por fuente")
plt.show()

df["country"].replace("", "Unknown").value_counts().head(15).plot(kind="bar")
plt.title("Top paises inferidos")
plt.show()

df.assign(date=df["timestamp"].dt.date).groupby(["date", "source"]).size().unstack(fill_value=0).plot()
plt.title("Eventos/noticias por dia y fuente")
plt.show()

sns.scatterplot(data=df.dropna(subset=["lat", "lon"]), x="lon", y="lat", hue="source")
plt.title("Puntos geograficos disponibles")
plt.show()
```

## Preparacion Para Logistic Regression, Naive Bayes Y KNN

Ejemplo con etiqueta manual o derivada. Para un proyecto serio, crea `label` revisando una muestra y definiendo clases como `escalation`, `diplomacy`, `military_activity`, `other`.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/dataset_integrado.csv")
df["text_ml"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
df = df[df["text_ml"] != ""].copy()

# Ejemplo didactico: reemplaza esto por etiquetas revisadas manualmente.
df["label"] = df["text_ml"].str.contains(
    "missile|attack|strike|war|military|nuclear",
    case=False,
    regex=True,
).map({True: "conflict_related", False: "other"})

X = df[["text_ml", "source", "country", "lat", "lon", "value"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "text_ml"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["source", "country"]),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]), ["lat", "lon", "value"]),
    ]
)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "naive_bayes": MultinomialNB(),
    "knn": KNeighborsClassifier(n_neighbors=5),
}

for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print("\\n", name)
    print(classification_report(y_test, pred))
```

Notas:

- Logistic Regression funciona muy bien con TF-IDF.
- Multinomial Naive Bayes requiere variables no negativas; TF-IDF y one-hot cumplen, pero cuidado con numericas escaladas. Si falla, usa solo texto/categoricas para NB.
- KNN necesita escalado y suele sufrir con alta dimensionalidad TF-IDF; prueba `TruncatedSVD` antes de KNN si el dataset crece.

## Fuentes De Documentacion Consultadas

- GDELT DOC 2.0 API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- BBC RSS feeds: https://support.bbc.co.uk/platform/feeds/NewsFeeds.htm
- BBC World RSS: https://feeds.bbci.co.uk/news/10628494
- Al Jazeera RSS usado: https://www.aljazeera.com/xml/rss/all.xml
- Google News RSS usado: https://news.google.com/rss/search?q=Iran+Israel+conflict
- OpenSky REST API: https://openskynetwork.github.io/opensky-api/rest.html
- NASA FIRMS Area API: https://firms.modaps.eosdis.nasa.gov/api/area/
- HTTP 429 y Retry-After: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/429
