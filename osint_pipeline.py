"""
Pipeline OSINT para un proyecto de ML sobre conflicto Iran-Israel-EE.UU.

El modulo esta disenado para dos usos:
1. Importarlo desde Jupyter Notebook.
2. Ejecutarlo como script: python osint_pipeline.py

No usa sys.exit(); si una fuente falla, registra el error y continua.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus, urljoin

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from requests import Response, Session
from requests.adapters import HTTPAdapter


DEFAULT_COLUMNS = [
    "timestamp",
    "source",
    "title",
    "text",
    "url",
    "country",
    "lat",
    "lon",
    "value",
]

OPTIONAL_COLUMNS = [
    "confidence",
    "satellite",
]

SOURCE_OUTPUTS = {
    "gdelt": "gdelt.csv",
    "bbc_rss": "bbc_rss.csv",
    "aljazeera_rss": "aljazeera_rss.csv",
    "google_news_rss": "google_news_rss.csv",
    "opensky": "opensky.csv",
    "nasa_firms": "nasa_firms.csv",
    "acled": "acled.csv",
    "worldpop": "worldpop.csv",
    "unhcr": "unhcr.csv",
    "hdx": "hdx.csv",
    "sentinel_hub": "sentinel_hub.csv",
    "openstreetmap": "openstreetmap.csv",
    "google_earth_engine": "google_earth_engine.csv",
    "ukmto": "ukmto.csv",
    "bluesky": "bluesky_posts.csv",
    "youtube": "youtube_metadata.csv",
}

DEFAULT_ENABLED_SOURCES = {
    "gdelt": True,
    "bbc_rss": True,
    "aljazeera_rss": True,
    "google_news_rss": True,
    "opensky": True,
    "nasa_firms": True,
    "acled": True,
    "worldpop": True,
    "unhcr": True,
    "hdx": True,
    "sentinel_hub": True,
    "openstreetmap": True,
    "google_earth_engine": False,
    "ukmto": True,
    "bluesky": True,
    "youtube": True,
}

DEFAULT_GOOGLE_NEWS_QUERIES = [
    "Iran Israel conflict",
    "Iran Israel escalation",
    "Iran US Israel Middle East",
]

DEFAULT_ACLED_KEYWORDS = [
    "Iran",
    "Israel",
    "Middle East",
    "Red Sea",
    "Gulf",
]

DEFAULT_ACLED_COUNTRIES = [
    "Iran",
    "Israel",
    "Palestine",
    "Lebanon",
    "Syria",
    "Iraq",
    "Yemen",
    "Saudi Arabia",
    "United Arab Emirates",
    "Qatar",
    "Oman",
    "Bahrain",
    "Kuwait",
]

DEFAULT_REGION_ISO3 = {
    "Iran": "IRN",
    "Israel": "ISR",
    "Palestine": "PSE",
    "Palestinian Territory": "PSE",
    "Lebanon": "LBN",
    "Syria": "SYR",
    "Iraq": "IRQ",
    "Yemen": "YEM",
    "Saudi Arabia": "SAU",
    "United Arab Emirates": "ARE",
    "Qatar": "QAT",
    "Oman": "OMN",
    "Bahrain": "BHR",
    "Kuwait": "KWT",
}

DEFAULT_WORLDPOP_COUNTRIES = [
    "IRN",
    "ISR",
    "PSE",
    "LBN",
    "SYR",
    "IRQ",
    "YEM",
]

DEFAULT_UNHCR_COUNTRIES = [
    "IRN",
    "ISR",
    "PSE",
    "LBN",
    "SYR",
    "IRQ",
    "YEM",
]

DEFAULT_HDX_QUERIES = [
    "Middle East hospitals",
    "Iran infrastructure",
    "Israel infrastructure",
    "Palestine humanitarian access",
    "Lebanon roads",
    "Syria camps",
    "Yemen humanitarian corridors",
]

DEFAULT_OSM_TAGS = {
    "highways": '["highway"]',
    "hospitals": '["amenity"="hospital"]',
    "clinics": '["amenity"="clinic"]',
    "airports": '["aeroway"="aerodrome"]',
    "ports": '["harbour"]',
    "power": '["power"]',
    "camps": '["tourism"="camp_site"]',
}

DEFAULT_BLUESKY_QUERIES = [
    "Iran Israel",
    "missiles Iran Israel",
    "Red Sea attacks",
    "Gaza escalation",
    "Middle East escalation",
]

DEFAULT_YOUTUBE_QUERIES = [
    "geopolitical analysis Iran Israel conflict",
    "Iran Israel OSINT",
    "Red Sea attacks",
    "Middle East escalation",
    "Iran Israel missiles analysis",
]

NEWS_SOURCES = {
    "gdelt",
    "bbc_rss",
    "aljazeera_rss",
    "google_news_rss",
    "bluesky",
    "youtube",
}

COUNTRY_COORDS = {
    "Iran": (32.4279, 53.6880),
    "Israel": (31.0461, 34.8516),
    "United States": (37.0902, -95.7129),
    "Palestinian Territory": (31.9522, 35.2332),
    "Lebanon": (33.8547, 35.8623),
    "Syria": (34.8021, 38.9968),
    "Iraq": (33.2232, 43.6793),
    "Jordan": (30.5852, 36.2384),
    "Yemen": (15.5527, 48.5164),
    "Qatar": (25.3548, 51.1839),
    "Saudi Arabia": (23.8859, 45.0792),
    "United Arab Emirates": (23.4241, 53.8478),
    "Oman": (21.4735, 55.9754),
    "Bahrain": (26.0667, 50.5577),
    "Kuwait": (29.3117, 47.4818),
}

LOCATION_COORDS = {
    "red sea": (20.2802, 38.5126),
    "gulf of oman": (24.6571, 58.0399),
    "strait of hormuz": (26.5667, 56.25),
    "hormuz": (26.5667, 56.25),
    "arabian gulf": (26.75, 51.0),
    "persian gulf": (26.75, 51.0),
    "aden": (12.7855, 45.0187),
    "mina saqr": (25.9707, 56.0528),
    "ras al khaymah": (25.8007, 55.9762),
    "ra's al khaymah": (25.8007, 55.9762),
    "al basrah": (30.5085, 47.7804),
    "basrah": (30.5085, 47.7804),
    "jubail": (27.0046, 49.6460),
    "khawr fakkan": (25.3313, 56.3420),
    "khor fakkan": (25.3313, 56.3420),
    "fujairah": (25.1288, 56.3265),
    "doha": (25.2854, 51.5310),
    "oman": (21.4735, 55.9754),
    "yemen": (15.5527, 48.5164),
    "uae": (23.4241, 53.8478),
    "united arab emirates": (23.4241, 53.8478),
    "saudi arabia": (23.8859, 45.0792),
    "qatar": (25.3548, 51.1839),
    "iraq": (33.2232, 43.6793),
}

COUNTRY_PATTERNS = [
    ("Iran", r"\b(iran|iranian|tehran)\b"),
    ("Israel", r"\b(israel|israeli|tel aviv|jerusalem|haifa)\b"),
    ("United States", r"\b(united states|u\.s\.|usa|us\b|american|washington)\b"),
    ("Palestinian Territory", r"\b(gaza|palestine|palestinian|west bank|hamas)\b"),
    ("Lebanon", r"\b(lebanon|lebanese|beirut|hezbollah)\b"),
    ("Syria", r"\b(syria|syrian|damascus)\b"),
    ("Iraq", r"\b(iraq|iraqi|baghdad)\b"),
    ("Jordan", r"\b(jordan|amman)\b"),
    ("Yemen", r"\b(yemen|houthi|houthis|sanaa)\b"),
    ("Qatar", r"\b(qatar|doha)\b"),
    ("Saudi Arabia", r"\b(saudi arabia|saudi|riyadh)\b"),
]


@dataclass
class PipelineConfig:
    """Configuracion central del pipeline."""

    data_dir: Path = Path("data")
    enabled_sources: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_ENABLED_SOURCES))
    request_timeout: int = 30
    max_retries: int = 4
    backoff_factor: float = 2.0
    backoff_max_seconds: float = 300.0
    cache_ttl_minutes: int = 60
    use_cache: bool = True
    user_agent: str = (
        "ProyectoFinalML1-OSINTPipeline/1.0 "
        "(academic research; contact: estudiante-universidad@example.com)"
    )
    gdelt_query: str = (
        '(Iran OR Iranian OR Israel OR Israeli OR Tehran OR "Tel Aviv" OR Jerusalem) '
        "(conflict OR escalation OR attack OR attacks OR strike OR missile OR "
        'military OR nuclear OR diplomacy OR diplomatic OR sanctions OR "United States" '
        "OR USA OR Washington)"
    )
    gdelt_timespan: str = "24h"
    gdelt_maxrecords: int = 100
    gdelt_sort: str = "datedesc"
    gdelt_min_interval_seconds: float = 30.0
    bbc_feed_url: str = "https://feeds.bbci.co.uk/news/world/rss.xml"
    aljazeera_feed_url: str = "https://www.aljazeera.com/xml/rss/all.xml"
    google_news_queries: list[str] = field(
        default_factory=lambda: list(DEFAULT_GOOGLE_NEWS_QUERIES)
    )
    google_news_hl: str = "en-US"
    google_news_gl: str = "US"
    google_news_ceid: str = "US:en"
    google_news_min_interval_seconds: float = 5.0
    opensky_bbox: tuple[float, float, float, float] = (24.0, 34.0, 40.0, 64.0)
    opensky_client_id: str | None = None
    opensky_client_secret: str | None = None
    nasa_firms_map_key: str | None = None
    nasa_firms_source: str = "VIIRS_SNPP_NRT"
    nasa_firms_area: str = "20,10,80,50"
    nasa_firms_day_range: int = 7
    nasa_firms_empty_retry_day_range: int | None = 14
    nasa_firms_date: str | None = None
    acled_api_key: str | None = None
    acled_username: str | None = None
    acled_password: str | None = None
    acled_days: int = 90
    acled_limit: int = 500
    acled_countries: list[str] = field(default_factory=lambda: list(DEFAULT_ACLED_COUNTRIES))
    acled_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_ACLED_KEYWORDS))
    worldpop_dataset: str = "wpgp"
    worldpop_year: int = 2020
    worldpop_countries: list[str] = field(default_factory=lambda: list(DEFAULT_WORLDPOP_COUNTRIES))
    worldpop_min_interval_seconds: float = 2.0
    unhcr_year: int = datetime.now(timezone.utc).year - 1
    unhcr_countries: list[str] = field(default_factory=lambda: list(DEFAULT_UNHCR_COUNTRIES))
    hdx_queries: list[str] = field(default_factory=lambda: list(DEFAULT_HDX_QUERIES))
    hdx_rows: int = 10
    hdx_min_interval_seconds: float = 2.0
    sentinel_hub_client_id: str | None = None
    sentinel_hub_client_secret: str | None = None
    sentinel_hub_collection: str = "sentinel-2-l2a"
    sentinel_hub_bbox: str = "34,24,64,40"
    sentinel_hub_days: int = 30
    sentinel_hub_limit: int = 50
    osm_bbox: str = "24,34,40,64"
    osm_tags: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_OSM_TAGS))
    osm_overpass_urls: list[str] = field(
        default_factory=lambda: [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
        ]
    )
    osm_min_interval_seconds: float = 5.0
    gee_project: str | None = None
    gee_collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    gee_bbox: str = "34,24,64,40"
    gee_days: int = 30
    ukmto_pages: list[str] = field(
        default_factory=lambda: [
            "https://www.ukmto.org/recent-incidents",
            "https://www.ukmto.org/ukmto-products/warnings",
            "https://www.ukmto.org/ukmto-products/advisories",
        ]
    )
    ukmto_max_reports: int = 25
    bluesky_queries: list[str] = field(default_factory=lambda: list(DEFAULT_BLUESKY_QUERIES))
    bluesky_limit: int = 50
    bluesky_days: int = 14
    bluesky_handle: str | None = None
    bluesky_app_password: str | None = None
    youtube_api_key: str | None = None
    youtube_queries: list[str] = field(default_factory=lambda: list(DEFAULT_YOUTUBE_QUERIES))
    youtube_max_results: int = 25
    youtube_days: int = 30

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Construye la configuracion desde variables de entorno y .env."""

        env_path = find_dotenv(usecwd=True)
        load_dotenv(env_path if env_path else None)
        config = cls()
        config.data_dir = Path(os.getenv("DATA_DIR", str(config.data_dir)))
        config.request_timeout = int(os.getenv("REQUEST_TIMEOUT", config.request_timeout))
        config.max_retries = int(os.getenv("MAX_RETRIES", config.max_retries))
        config.backoff_factor = float(os.getenv("BACKOFF_FACTOR", config.backoff_factor))
        config.backoff_max_seconds = float(
            os.getenv("BACKOFF_MAX_SECONDS", config.backoff_max_seconds)
        )
        config.cache_ttl_minutes = int(os.getenv("CACHE_TTL_MINUTES", config.cache_ttl_minutes))
        config.use_cache = parse_bool(os.getenv("USE_CACHE"), config.use_cache)
        config.user_agent = os.getenv("USER_AGENT", config.user_agent)
        config.gdelt_query = os.getenv("GDELT_QUERY", config.gdelt_query)
        config.gdelt_timespan = os.getenv("GDELT_TIMESPAN", config.gdelt_timespan)
        config.gdelt_maxrecords = int(os.getenv("GDELT_MAXRECORDS", config.gdelt_maxrecords))
        config.gdelt_sort = os.getenv("GDELT_SORT", config.gdelt_sort)
        config.gdelt_min_interval_seconds = float(
            os.getenv("GDELT_MIN_INTERVAL_SECONDS", config.gdelt_min_interval_seconds)
        )
        config.bbc_feed_url = os.getenv("BBC_FEED_URL", config.bbc_feed_url)
        config.aljazeera_feed_url = os.getenv("ALJAZEERA_FEED_URL", config.aljazeera_feed_url)
        config.google_news_queries = split_query_list(
            os.getenv("GOOGLE_NEWS_QUERIES"),
            config.google_news_queries,
        )
        config.google_news_hl = os.getenv("GOOGLE_NEWS_HL", config.google_news_hl)
        config.google_news_gl = os.getenv("GOOGLE_NEWS_GL", config.google_news_gl)
        config.google_news_ceid = os.getenv("GOOGLE_NEWS_CEID", config.google_news_ceid)
        config.google_news_min_interval_seconds = float(
            os.getenv(
                "GOOGLE_NEWS_MIN_INTERVAL_SECONDS",
                config.google_news_min_interval_seconds,
            )
        )
        config.opensky_bbox = parse_bbox(os.getenv("OPENSKY_BBOX"), config.opensky_bbox)
        config.opensky_client_id = os.getenv("OPENSKY_CLIENT_ID") or None
        config.opensky_client_secret = os.getenv("OPENSKY_CLIENT_SECRET") or None
        config.nasa_firms_map_key = (os.getenv("NASA_FIRMS_MAP_KEY") or "").strip() or None
        config.nasa_firms_source = os.getenv("NASA_FIRMS_SOURCE", config.nasa_firms_source)
        config.nasa_firms_area = normalize_nasa_area(
            os.getenv("NASA_FIRMS_AREA"),
            config.nasa_firms_area,
        )
        config.nasa_firms_day_range = int(
            os.getenv("NASA_FIRMS_DAY_RANGE", config.nasa_firms_day_range)
        )
        config.nasa_firms_empty_retry_day_range = parse_optional_int(
            os.getenv("NASA_FIRMS_EMPTY_RETRY_DAY_RANGE"),
            config.nasa_firms_empty_retry_day_range,
        )
        config.nasa_firms_date = os.getenv("NASA_FIRMS_DATE") or None
        config.acled_api_key = (
            os.getenv("ACLED_API_KEY")
            or os.getenv("ACLED_ACCESS_TOKEN")
            or ""
        ).strip() or None
        config.acled_username = (
            os.getenv("ACLED_USERNAME")
            or os.getenv("ACLED_EMAIL")
            or ""
        ).strip() or None
        config.acled_password = (os.getenv("ACLED_PASSWORD") or "").strip() or None
        config.acled_days = int(os.getenv("ACLED_DAYS", config.acled_days))
        config.acled_limit = int(os.getenv("ACLED_LIMIT", config.acled_limit))
        config.acled_countries = split_query_list(
            os.getenv("ACLED_COUNTRIES"),
            config.acled_countries,
        )
        config.acled_keywords = split_query_list(
            os.getenv("ACLED_KEYWORDS"),
            config.acled_keywords,
        )
        config.worldpop_dataset = os.getenv("WORLDPOP_DATASET", config.worldpop_dataset)
        config.worldpop_year = int(os.getenv("WORLDPOP_YEAR", config.worldpop_year))
        config.worldpop_countries = split_query_list(
            os.getenv("WORLDPOP_COUNTRIES"),
            config.worldpop_countries,
        )
        config.worldpop_min_interval_seconds = float(
            os.getenv("WORLDPOP_MIN_INTERVAL_SECONDS", config.worldpop_min_interval_seconds)
        )
        config.unhcr_year = int(os.getenv("UNHCR_YEAR", config.unhcr_year))
        config.unhcr_countries = split_query_list(
            os.getenv("UNHCR_COUNTRIES"),
            config.unhcr_countries,
        )
        config.hdx_queries = split_query_list(os.getenv("HDX_QUERIES"), config.hdx_queries)
        config.hdx_rows = int(os.getenv("HDX_ROWS", config.hdx_rows))
        config.hdx_min_interval_seconds = float(
            os.getenv("HDX_MIN_INTERVAL_SECONDS", config.hdx_min_interval_seconds)
        )
        config.sentinel_hub_client_id = (os.getenv("SENTINEL_HUB_CLIENT_ID") or "").strip() or None
        config.sentinel_hub_client_secret = (
            os.getenv("SENTINEL_HUB_CLIENT_SECRET") or ""
        ).strip() or None
        config.sentinel_hub_collection = os.getenv(
            "SENTINEL_HUB_COLLECTION",
            config.sentinel_hub_collection,
        )
        config.sentinel_hub_bbox = normalize_lonlat_bbox(
            os.getenv("SENTINEL_HUB_BBOX"),
            config.sentinel_hub_bbox,
        )
        config.sentinel_hub_days = int(os.getenv("SENTINEL_HUB_DAYS", config.sentinel_hub_days))
        config.sentinel_hub_limit = int(os.getenv("SENTINEL_HUB_LIMIT", config.sentinel_hub_limit))
        config.osm_bbox = normalize_latlon_bbox(os.getenv("OSM_BBOX"), config.osm_bbox)
        config.osm_overpass_urls = split_query_list(
            os.getenv("OSM_OVERPASS_URLS"),
            config.osm_overpass_urls,
        )
        config.osm_min_interval_seconds = float(
            os.getenv("OSM_MIN_INTERVAL_SECONDS", config.osm_min_interval_seconds)
        )
        config.gee_project = (os.getenv("GEE_PROJECT") or "").strip() or None
        config.gee_collection = os.getenv("GEE_COLLECTION", config.gee_collection)
        config.gee_bbox = normalize_lonlat_bbox(os.getenv("GEE_BBOX"), config.gee_bbox)
        config.gee_days = int(os.getenv("GEE_DAYS", config.gee_days))
        config.ukmto_pages = split_query_list(os.getenv("UKMTO_PAGES"), config.ukmto_pages)
        config.ukmto_max_reports = int(os.getenv("UKMTO_MAX_REPORTS", config.ukmto_max_reports))
        config.bluesky_queries = split_query_list(
            os.getenv("BLUESKY_QUERIES"),
            config.bluesky_queries,
        )
        config.bluesky_limit = int(os.getenv("BLUESKY_LIMIT", config.bluesky_limit))
        config.bluesky_days = int(os.getenv("BLUESKY_DAYS", config.bluesky_days))
        config.bluesky_handle = (
            os.getenv("BLUESKY_HANDLE")
            or os.getenv("BSKY_HANDLE")
            or ""
        ).strip() or None
        config.bluesky_app_password = (
            os.getenv("BLUESKY_APP_PASSWORD")
            or os.getenv("BSKY_APP_PASSWORD")
            or ""
        ).strip() or None
        config.youtube_api_key = (os.getenv("YOUTUBE_API_KEY") or "").strip() or None
        config.youtube_queries = split_query_list(
            os.getenv("YOUTUBE_QUERIES"),
            config.youtube_queries,
        )
        config.youtube_max_results = int(
            os.getenv("YOUTUBE_MAX_RESULTS", config.youtube_max_results)
        )
        config.youtube_days = int(os.getenv("YOUTUBE_DAYS", config.youtube_days))

        enabled_from_env = os.getenv("ENABLED_SOURCES")
        disabled_from_env = os.getenv("DISABLED_SOURCES")
        if enabled_from_env:
            config.enabled_sources = sources_from_csv(enabled_from_env)
        if disabled_from_env:
            for source_name in split_csv(disabled_from_env):
                if source_name in config.enabled_sources:
                    config.enabled_sources[source_name] = False

        return config


class RateLimiter:
    """Rate limiter simple por clave, suficiente para llamadas secuenciales."""

    def __init__(self) -> None:
        self._last_call: dict[str, float] = {}

    def wait(self, key: str, min_interval_seconds: float) -> None:
        """Espera si la ultima llamada para la clave fue demasiado reciente."""

        if min_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_call.get(key, 0.0)
        wait_seconds = min_interval_seconds - elapsed
        if wait_seconds > 0:
            wait_seconds += random.uniform(0.25, 1.25)
            logging.info("Rate limit local para %s: esperando %.1f segundos", key, wait_seconds)
            time.sleep(wait_seconds)
        self._last_call[key] = time.monotonic()


class OpenSkyTokenManager:
    """Gestiona OAuth2 client credentials para OpenSky cuando hay credenciales."""

    TOKEN_URL = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/"
        "protocol/openid-connect/token"
    )

    def __init__(
        self,
        session: Session,
        client_id: str | None,
        client_secret: str | None,
        config: PipelineConfig,
    ) -> None:
        self.session = session
        self.client_id = client_id
        self.client_secret = client_secret
        self.config = config
        self.token: str | None = None
        self.expires_at: datetime | None = None

    @property
    def enabled(self) -> bool:
        """Indica si hay credenciales OAuth disponibles."""

        return bool(self.client_id and self.client_secret)

    def headers(self) -> dict[str, str]:
        """Devuelve headers con Bearer token o dict vacio si no hay credenciales."""

        if not self.enabled:
            return {}
        token = self.get_token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def get_token(self) -> str | None:
        """Devuelve un token vigente y lo refresca cerca del vencimiento."""

        if self.token and self.expires_at and datetime.now(timezone.utc) < self.expires_at:
            return self.token
        return self.refresh_token()

    def refresh_token(self) -> str | None:
        """Solicita un token nuevo a OpenSky sin interrumpir el pipeline si falla."""

        if not self.enabled:
            return None
        response = request_with_retries(
            self.session,
            "POST",
            self.TOKEN_URL,
            config=self.config,
            source_name="opensky_auth",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response is None:
            logging.warning("No fue posible obtener token OAuth de OpenSky; se usara acceso anonimo.")
            return None
        payload = safe_json(response, "opensky_auth")
        if not payload or "access_token" not in payload:
            logging.warning("Respuesta OAuth de OpenSky invalida; se usara acceso anonimo.")
            return None
        expires_in = int(payload.get("expires_in", 1800))
        self.token = payload["access_token"]
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(expires_in - 30, 60))
        return self.token


class ACLEDTokenManager:
    """Gestiona autenticacion ACLED por token directo u OAuth password grant."""

    TOKEN_URL = "https://acleddata.com/oauth/token"

    def __init__(self, session: Session, config: PipelineConfig) -> None:
        self.session = session
        self.config = config
        self.token: str | None = config.acled_api_key
        self.expires_at: datetime | None = None

    def headers(self) -> dict[str, str] | None:
        """Devuelve headers Authorization o None si no hay credenciales."""

        token = self.get_token()
        if not token:
            return None
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def get_token(self) -> str | None:
        """Usa ACLED_API_KEY como token o solicita uno con usuario/password."""

        if self.token and self.config.acled_api_key:
            return self.token
        if self.token and self.expires_at and datetime.now(timezone.utc) < self.expires_at:
            return self.token
        if not (self.config.acled_username and self.config.acled_password):
            return None
        return self.refresh_token()

    def refresh_token(self) -> str | None:
        """Solicita token OAuth ACLED sin detener el pipeline si falla."""

        response = request_with_retries(
            self.session,
            "POST",
            self.TOKEN_URL,
            config=self.config,
            source_name="acled_auth",
            data={
                "username": self.config.acled_username,
                "password": self.config.acled_password,
                "grant_type": "password",
                "client_id": "acled",
                "scope": "authenticated",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response is None:
            logging.warning("No fue posible autenticar ACLED; se omitira esta fuente.")
            return None
        payload = safe_json(response, "acled_auth")
        if not isinstance(payload, dict) or "access_token" not in payload:
            logging.warning("Respuesta OAuth ACLED invalida; se omitira esta fuente.")
            return None
        expires_in = int(payload.get("expires_in", 86400))
        self.token = str(payload["access_token"])
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(expires_in - 60, 60))
        return self.token


def parse_bool(raw_value: str | None, default: bool) -> bool:
    """Convierte texto de entorno en booleano."""

    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def split_csv(raw_value: str) -> list[str]:
    """Parte listas separadas por coma y normaliza espacios."""

    return [item.strip() for item in raw_value.split(",") if item.strip()]


def split_query_list(raw_value: str | None, default: list[str]) -> list[str]:
    """Parte queries RSS separadas por punto y coma o barra vertical."""

    if not raw_value:
        return list(default)
    queries = [item.strip() for item in re.split(r"[;|]", raw_value) if item.strip()]
    return queries or list(default)


def parse_optional_int(raw_value: str | None, default: int | None) -> int | None:
    """Parsea enteros opcionales desde entorno, aceptando vacio/none/null."""

    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if not value or value in {"none", "null", "false", "0"}:
        return None
    try:
        return int(value)
    except ValueError:
        logging.warning("Valor entero opcional invalido: %s. Se usara %s.", raw_value, default)
        return default


def sources_from_csv(raw_value: str) -> dict[str, bool]:
    """Crea el mapa enabled_sources a partir de una lista de fuentes."""

    selected = set(split_csv(raw_value))
    return {source_name: source_name in selected for source_name in DEFAULT_ENABLED_SOURCES}


def parse_bbox(
    raw_value: str | None,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Parsea bounding boxes lat_min,lon_min,lat_max,lon_max."""

    if not raw_value:
        return default
    try:
        values = tuple(float(part.strip()) for part in raw_value.split(","))
    except ValueError:
        logging.warning("OPENSKY_BBOX invalido: %s. Se usara el valor por defecto.", raw_value)
        return default
    if len(values) != 4:
        logging.warning("OPENSKY_BBOX requiere 4 valores. Se usara el valor por defecto.")
        return default
    return values  # type: ignore[return-value]


def normalize_latlon_bbox(raw_value: str | None, default: str) -> str:
    """Valida bbox lat_min,lon_min,lat_max,lon_max y devuelve texto estable."""

    if not raw_value:
        return default
    try:
        values = [float(part.strip()) for part in raw_value.split(",")]
    except ValueError:
        logging.warning("BBox lat/lon invalido: %s. Se usara %s.", raw_value, default)
        return default
    if len(values) != 4:
        logging.warning("BBox lat/lon requiere 4 valores. Se usara %s.", default)
        return default
    lat_min, lon_min, lat_max, lon_max = values
    if lat_min >= lat_max or lon_min >= lon_max:
        logging.warning("BBox lat/lon debe ser lat_min,lon_min,lat_max,lon_max. Se usara %s.", default)
        return default
    return ",".join(format_float(value) for value in values)


def normalize_lonlat_bbox(raw_value: str | None, default: str) -> str:
    """Valida bbox west,south,east,north y devuelve texto estable."""

    if not raw_value:
        return default
    try:
        values = [float(part.strip()) for part in raw_value.split(",")]
    except ValueError:
        logging.warning("BBox lon/lat invalido: %s. Se usara %s.", raw_value, default)
        return default
    if len(values) != 4:
        logging.warning("BBox lon/lat requiere 4 valores. Se usara %s.", default)
        return default
    west, south, east, north = values
    if west >= east or south >= north:
        logging.warning("BBox lon/lat debe ser west,south,east,north. Se usara %s.", default)
        return default
    return ",".join(format_float(value) for value in values)


def bbox_center_from_lonlat(bbox_text: str) -> tuple[float | None, float | None]:
    """Centroide simple de bbox west,south,east,north."""

    try:
        west, south, east, north = [float(part.strip()) for part in bbox_text.split(",")]
    except ValueError:
        return None, None
    return (south + north) / 2, (west + east) / 2


def lonlat_bbox_to_overpass(bbox_text: str) -> str:
    """Convierte lat_min,lon_min,lat_max,lon_max al orden Overpass south,west,north,east."""

    lat_min, lon_min, lat_max, lon_max = [float(part.strip()) for part in bbox_text.split(",")]
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"


def iso3_to_country_name(iso3: str) -> str:
    """Mapea ISO3 regional a nombre legible."""

    clean = iso3.strip().upper()
    for country, code in DEFAULT_REGION_ISO3.items():
        if code == clean:
            return country
    return clean


def normalize_nasa_area(raw_value: str | None, default: str) -> str:
    """Valida bbox FIRMS en formato west,south,east,north y devuelve texto estable."""

    if not raw_value:
        return default
    try:
        values = [float(part.strip()) for part in raw_value.split(",")]
    except ValueError:
        logging.warning("NASA_FIRMS_AREA invalido: %s. Se usara %s.", raw_value, default)
        return default
    if len(values) != 4:
        logging.warning("NASA_FIRMS_AREA requiere 4 valores. Se usara %s.", default)
        return default
    west, south, east, north = values
    if west >= east or south >= north:
        logging.warning(
            "NASA_FIRMS_AREA debe ser west,south,east,north. Se usara %s.",
            default,
        )
        return default
    return ",".join(format_float(value) for value in values)


def format_float(value: float) -> str:
    """Formatea floats sin ceros innecesarios para URLs reproducibles."""

    return f"{value:g}"


def configure_logging(level: str = "INFO") -> None:
    """Configura logging claro para consola y notebooks."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def build_session(config: PipelineConfig) -> Session:
    """Crea una requests.Session reutilizable con headers conservadores."""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.user_agent,
            "Accept": "application/json, text/csv, application/rss+xml, application/xml, text/xml, */*",
            "Connection": "keep-alive",
        }
    )
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def parse_retry_after(header_value: str | None) -> float | None:
    """Interpreta Retry-After en segundos o fecha HTTP."""

    if not header_value:
        return None
    value = header_value.strip()
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def compute_sleep_seconds(
    attempt: int,
    response: Response | None,
    config: PipelineConfig,
) -> float:
    """Calcula exponential backoff con jitter y respeta Retry-After si existe."""

    retry_after = parse_retry_after(response.headers.get("Retry-After") if response else None)
    if retry_after is not None:
        return retry_after
    sleep_seconds = config.backoff_factor * (2 ** max(attempt - 1, 0))
    sleep_seconds += random.uniform(0.0, 1.0)
    return min(sleep_seconds, config.backoff_max_seconds)


def is_retryable_status(status_code: int) -> bool:
    """Define HTTP status temporales que ameritan reintento."""

    return status_code == 429 or 500 <= status_code <= 599


def request_with_retries(
    session: Session,
    method: str,
    url: str,
    config: PipelineConfig,
    source_name: str,
    return_response_on_statuses: set[int] | None = None,
    **kwargs: Any,
) -> Response | None:
    """Ejecuta una peticion HTTP robusta con retries, backoff y manejo de 429."""

    timeout = kwargs.pop("timeout", config.request_timeout)
    for attempt in range(1, config.max_retries + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            logging.warning(
                "%s intento %s/%s fallo por conexion: %s",
                source_name,
                attempt,
                config.max_retries,
                exc,
            )
            if attempt == config.max_retries:
                return None
            sleep_seconds = compute_sleep_seconds(attempt, None, config)
            logging.info("%s reintentara en %.1f segundos", source_name, sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        if 200 <= response.status_code < 300:
            return response

        retryable = is_retryable_status(response.status_code)
        body_preview = response.text[:250].replace("\n", " ") if response.text else ""
        logging.warning(
            "%s recibio HTTP %s en intento %s/%s. Respuesta: %s",
            source_name,
            response.status_code,
            attempt,
            config.max_retries,
            body_preview,
        )

        if not retryable or attempt == config.max_retries:
            if return_response_on_statuses and response.status_code in return_response_on_statuses:
                return response
            return None

        sleep_seconds = compute_sleep_seconds(attempt, response, config)
        if response.status_code == 429:
            logging.warning(
                "%s alcanzo rate limit HTTP 429. Esperando %.1f segundos antes de reintentar.",
                source_name,
                sleep_seconds,
            )
        else:
            logging.info("%s reintentara en %.1f segundos", source_name, sleep_seconds)
        time.sleep(sleep_seconds)

    return None


def safe_json(response: Response, source_name: str) -> dict[str, Any] | list[Any] | None:
    """Valida que una respuesta sea JSON antes de parsearla."""

    content_type = response.headers.get("Content-Type", "").lower()
    body = response.text.lstrip()
    looks_like_json = body.startswith("{") or body.startswith("[")
    if "json" not in content_type and not looks_like_json:
        logging.warning(
            "%s devolvio Content-Type no JSON (%s). Se ignora la respuesta.",
            source_name,
            content_type or "sin Content-Type",
        )
        return None
    try:
        return response.json()
    except ValueError as exc:
        logging.warning("%s devolvio JSON invalido: %s", source_name, exc)
        return None


def strip_html(value: Any) -> str:
    """Limpia HTML basico de summaries RSS."""

    if value is None:
        return ""
    text = str(value)
    if "<" not in text or ">" not in text:
        return text.strip()
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def parse_any_timestamp(value: Any) -> pd.Timestamp:
    """Convierte fechas heterogeneas a UTC; devuelve NaT si no puede."""

    if value is None or value == "":
        return pd.NaT
    if isinstance(value, (int, float)) and not pd.isna(value):
        return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def parse_int(value: Any, default: int = 0) -> int:
    """Convierte valores numericos heterogeneos a int tolerante."""

    if value is None or value == "" or pd.isna(value):
        return default
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def iso_utc_days_ago(days: int) -> str:
    """Fecha ISO UTC para parametros since/publishedAfter."""

    start = datetime.now(timezone.utc) - timedelta(days=max(int(days), 1))
    return start.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ymd_days_ago(days: int) -> str:
    """Fecha YYYY-MM-DD UTC para APIs que usan rangos de fechas simples."""

    start = datetime.now(timezone.utc) - timedelta(days=max(int(days), 1))
    return start.strftime("%Y-%m-%d")


def infer_country_from_text(title: str, text: str) -> str | None:
    """Infiere pais por palabras clave simples y transparentes."""

    haystack = f"{title} {text}".lower()
    for country, pattern in COUNTRY_PATTERNS:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return country
    return None


def infer_location_coords(text: str) -> tuple[float | None, float | None]:
    """Busca coordenadas aproximadas por lugares maritimos frecuentes."""

    haystack = text.lower()
    for key, coords in LOCATION_COORDS.items():
        if key in haystack:
            return coords
    country = infer_country_from_text("", text)
    return coords_for_country(country)


def parse_coordinate_pair(text: str) -> tuple[float | None, float | None]:
    """Extrae pares lat/lon en decimal o con hemisferio cuando aparecen en reportes."""

    if not text:
        return None, None

    decimal = re.search(
        r"(?P<lat>-?\d{1,2}\.\d+)\s*[,/ ]+\s*(?P<lon>-?\d{1,3}\.\d+)",
        text,
    )
    if decimal:
        return float(decimal.group("lat")), float(decimal.group("lon"))

    hemi = re.search(
        r"(?P<lat>\d{1,2}(?:\.\d+)?)\s*°?\s*(?P<lat_hemi>[NS])"
        r".{0,12}?"
        r"(?P<lon>\d{1,3}(?:\.\d+)?)\s*°?\s*(?P<lon_hemi>[EW])",
        text,
        flags=re.IGNORECASE,
    )
    if hemi:
        lat = float(hemi.group("lat"))
        lon = float(hemi.group("lon"))
        if hemi.group("lat_hemi").upper() == "S":
            lat = -lat
        if hemi.group("lon_hemi").upper() == "W":
            lon = -lon
        return lat, lon

    return infer_location_coords(text)


def detect_incident_type(text: str) -> str:
    """Clasifica incidentes maritimos por palabras clave transparentes."""

    haystack = text.lower()
    for label, pattern in [
        ("hijack", r"\bhijack|seiz"),
        ("boarding", r"\bboarding|boarded"),
        ("attack", r"\battack|projectile|missile|uav|drone|strike|fire\b"),
        ("suspicious_activity", r"suspicious|skiff|approach|hail"),
        ("electronic_interference", r"interference|jam|gps|ais"),
        ("advisory", r"advisory|notice"),
    ]:
        if re.search(pattern, haystack):
            return label
    return "maritime_security"


def extract_ukmto_location(text: str) -> str:
    """Extrae una frase corta de ubicacion desde reportes UKMTO."""

    patterns = [
        r"incident\s+(?:approximately\s+)?(?:\d+\s*NM\s+)?(?:[a-z]+\s+of\s+)?(?P<loc>[A-Z][A-Za-z'’\s,-]+?)(?:\.|\n| in TTW|,)",
        r"(?P<loc>Red Sea|Gulf of Oman|Strait of Hormuz|Arabian Gulf|Persian Gulf|Aden|Jubail|Al Basrah|Khawr Fakkan|Fujairah|Doha|Mina Saqr|Ra.?s al Khaymah)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group("loc")).strip(" ,.-")
    return ""


def extract_ukmto_timestamp(text: str, fallback_url: str = "") -> pd.Timestamp:
    """Extrae fecha/hora UKMTO desde el contenido o el nombre del PDF."""

    date_match = re.search(r"Report Date:\s*(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", text)
    time_match = re.search(r"Report Time:\s*(\d{3,4})\s*UTC", text, flags=re.IGNORECASE)
    if date_match:
        date_text = date_match.group(1)
        time_text = (time_match.group(1) if time_match else "0000").zfill(4)
        return parse_any_timestamp(f"{date_text} {time_text[:2]}:{time_text[2:]} UTC")

    url_date = re.search(r"(20\d{2})(\d{2})(\d{2})", fallback_url)
    if url_date:
        year, month, day = url_date.groups()
        return parse_any_timestamp(f"{year}-{month}-{day} 00:00 UTC")
    return pd.NaT


def parse_pdf_text(content: bytes, source_name: str) -> str:
    """Extrae texto de PDF si pypdf esta instalado; si no, devuelve vacio."""

    try:
        from pypdf import PdfReader
    except ImportError:
        logging.warning("%s requiere pypdf para leer PDFs; instala requirements.txt actualizado.", source_name)
        return ""

    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        logging.warning("%s no pudo extraer texto PDF: %s", source_name, exc)
        return ""


def bluesky_post_url(uri: str, handle: str) -> str:
    """Construye URL navegable de un post Bluesky desde AT URI."""

    rkey = uri.rstrip("/").rsplit("/", 1)[-1] if uri else ""
    if handle and rkey:
        return f"https://bsky.app/profile/{handle}/post/{rkey}"
    return "https://bsky.app/"


def acled_country_filter(countries: list[str]) -> str:
    """Construye filtro OR de paises para ACLED."""

    clean = [country.strip() for country in countries if country.strip()]
    if not clean:
        return ""
    first, *rest = clean
    return first + "".join(f":OR:country={country}" for country in rest)


def is_project_relevant(title: str, text: str) -> bool:
    """Evalua si una noticia mantiene foco Iran-Israel-EE.UU./Medio Oriente."""

    haystack = f"{title} {text}".lower()
    has_iran = bool(re.search(r"\b(iran|iranian|tehran)\b", haystack))
    has_israel = bool(
        re.search(r"\b(israel|israeli|tel aviv|jerusalem|haifa)\b", haystack)
    )
    has_us = bool(
        re.search(
            r"\b(united states|u\.s\.|usa|american|washington|white house|pentagon|trump)\b",
            haystack,
        )
    )
    has_regional_actor = bool(
        re.search(
            r"\b(middle east|gaza|palestine|lebanon|syria|iraq|yemen|houthi|hezbollah)\b",
            haystack,
        )
    )
    has_theme = bool(
        re.search(
            r"\b(conflict|escalation|attack|attacks|strike|strikes|missile|drone|"
            r"military|war|nuclear|diplomacy|diplomatic|ceasefire|sanctions|"
            r"talks|negotiation|retaliation|airspace)\b",
            haystack,
        )
    )
    has_core_actor = has_iran or has_israel
    return (has_core_actor and (has_theme or has_us or has_regional_actor)) or (
        has_us and has_regional_actor and has_theme
    )


def infer_country_from_coords(lat: float | None, lon: float | None) -> str | None:
    """Infiere pais por cajas geograficas aproximadas para el area de estudio."""

    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    if 24.0 <= lat <= 40.0 and 44.0 <= lon <= 64.0:
        return "Iran"
    if 29.0 <= lat <= 34.0 and 34.0 <= lon <= 36.0:
        return "Israel"
    if 24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0:
        return "United States"
    if 31.0 <= lat <= 33.5 and 34.0 <= lon <= 36.0:
        return "Palestinian Territory"
    if 33.0 <= lat <= 35.0 and 35.0 <= lon <= 37.0:
        return "Lebanon"
    if 32.0 <= lat <= 37.5 and 35.5 <= lon <= 42.5:
        return "Syria"
    if 29.0 <= lat <= 38.0 and 39.0 <= lon <= 49.0:
        return "Iraq"
    return None


def coords_for_country(country: str | None) -> tuple[float | None, float | None]:
    """Devuelve coordenadas centroides cuando no hay lat/lon puntuales."""

    if not country:
        return None, None
    return COUNTRY_COORDS.get(country, (None, None))


def ensure_normalized(df: pd.DataFrame | None, source_name: str) -> pd.DataFrame | None:
    """Garantiza columnas normalizadas, tipos tolerantes y orden estable."""

    if df is None:
        return None
    for column in DEFAULT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    output_columns = DEFAULT_COLUMNS + [column for column in OPTIONAL_COLUMNS if column in df.columns]
    normalized = df[output_columns].copy()
    normalized["source"] = normalized["source"].fillna(source_name)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    for column in ["title", "text", "url", "country"]:
        normalized[column] = normalized[column].fillna("").astype(str)
    normalized["lat"] = pd.to_numeric(normalized["lat"], errors="coerce")
    normalized["lon"] = pd.to_numeric(normalized["lon"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    for column in ["confidence", "satellite"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].fillna("").astype(str)
    return normalized


def filter_thematic_news(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Filtra noticias para sostener coherencia tematica del proyecto."""

    if source_name not in NEWS_SOURCES or df.empty:
        return df
    mask = df.apply(lambda row: is_project_relevant(row["title"], row["text"]), axis=1)
    filtered = df[mask].reset_index(drop=True)
    removed = len(df) - len(filtered)
    if removed:
        logging.info("%s filtro tematico removio %s filas fuera de foco.", source_name, removed)
    return filtered


def should_use_cache(path: Path, config: PipelineConfig) -> bool:
    """Determina si un CSV existente puede reutilizarse para bajar presion a APIs."""

    if not config.use_cache or not path.exists() or config.cache_ttl_minutes <= 0:
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= config.cache_ttl_minutes * 60


def load_cached_csv(path: Path, source_name: str) -> pd.DataFrame | None:
    """Carga cache local sin tumbar el pipeline si el CSV esta corrupto."""

    try:
        cached = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logging.warning("%s cache invalido en %s: %s", source_name, path, exc)
        return None
    logging.info("%s usando cache local: %s", source_name, path)
    normalized = ensure_normalized(cached, source_name)
    return filter_thematic_news(normalized, source_name) if normalized is not None else None


def load_available_source_csv(path: Path, source_name: str) -> pd.DataFrame | None:
    """Carga un CSV existente para integracion cuando una fuente no trajo datos nuevos."""

    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logging.warning("%s CSV disponible invalido en %s: %s", source_name, path, exc)
        return None
    logging.info("%s no produjo datos nuevos; se integrara CSV disponible: %s", source_name, path)
    normalized = ensure_normalized(cached, source_name)
    return filter_thematic_news(normalized, source_name) if normalized is not None else None


def save_csv(df: pd.DataFrame, path: Path, source_name: str) -> bool:
    """Guarda un DataFrame en CSV sin romper el flujo si hay error de disco."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        logging.info("%s guardado en %s (%s filas)", source_name, path, len(df))
        return True
    except OSError as exc:
        logging.error("%s no pudo guardarse en %s: %s", source_name, path, exc)
        return False


def fetch_gdelt(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta GDELT DOC 2.0 ArticleList con defensas contra 429."""

    source_name = "gdelt"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    rate_limiter.wait("gdelt", config.gdelt_min_interval_seconds)
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": config.gdelt_query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(max(config.gdelt_maxrecords, 1), 250),
        "timespan": config.gdelt_timespan,
        "sort": config.gdelt_sort,
    }
    headers = {
        "User-Agent": config.user_agent,
        "Accept": "application/json",
    }
    response = request_with_retries(
        session,
        "GET",
        url,
        config=config,
        source_name=source_name,
        params=params,
        headers=headers,
    )
    if response is None:
        logging.error("GDELT fallo o siguio limitado; el pipeline continuara sin esta fuente.")
        return None

    payload = safe_json(response, source_name)
    if not isinstance(payload, dict):
        return None

    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        logging.warning("GDELT JSON no contiene lista 'articles'.")
        return None

    rows: list[dict[str, Any]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = strip_html(article.get("title"))
        url_value = article.get("url") or article.get("url_mobile") or ""
        country = article.get("sourcecountry") or infer_country_from_text(title, "")
        lat, lon = coords_for_country(country)
        rows.append(
            {
                "timestamp": parse_any_timestamp(article.get("seendate")),
                "source": source_name,
                "title": title,
                "text": title,
                "url": url_value,
                "country": country or "",
                "lat": lat,
                "lon": lon,
                "value": pd.NA,
            }
        )

    df = ensure_normalized(pd.DataFrame(rows), source_name)
    if df is not None:
        df = filter_thematic_news(df, source_name)
        save_csv(df, cache_path, source_name)
    return df


def fetch_rss_feed(
    session: Session,
    config: PipelineConfig,
    source_name: str,
    feed_url: str,
) -> pd.DataFrame | None:
    """Descarga y normaliza un feed RSS/Atom."""

    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    response = request_with_retries(
        session,
        "GET",
        feed_url,
        config=config,
        source_name=source_name,
        headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"},
    )
    if response is None:
        logging.error("%s fallo; el pipeline continuara sin esta fuente.", source_name)
        return None

    parsed = feedparser.parse(response.content)
    if getattr(parsed, "bozo", False):
        logging.warning("%s RSS tiene advertencias de parsing: %s", source_name, parsed.bozo_exception)

    entries = getattr(parsed, "entries", [])
    rows: list[dict[str, Any]] = []
    for entry in entries:
        title = strip_html(entry.get("title", ""))
        text = strip_html(entry.get("summary", entry.get("description", "")))
        link = entry.get("link", "")
        published = entry.get("published") or entry.get("updated")
        country = infer_country_from_text(title, text)
        lat, lon = coords_for_country(country)
        rows.append(
            {
                "timestamp": parse_any_timestamp(published),
                "source": source_name,
                "title": title,
                "text": text,
                "url": link,
                "country": country or "",
                "lat": lat,
                "lon": lon,
                "value": pd.NA,
            }
        )

    df = ensure_normalized(pd.DataFrame(rows), source_name)
    if df is not None:
        df = filter_thematic_news(df, source_name)
        save_csv(df, cache_path, source_name)
    return df


def fetch_bbc_rss(session: Session, config: PipelineConfig, _: RateLimiter) -> pd.DataFrame | None:
    """Wrapper para BBC RSS."""

    return fetch_rss_feed(session, config, "bbc_rss", config.bbc_feed_url)


def fetch_aljazeera_rss(
    session: Session,
    config: PipelineConfig,
    _: RateLimiter,
) -> pd.DataFrame | None:
    """Wrapper para Al Jazeera RSS."""

    return fetch_rss_feed(session, config, "aljazeera_rss", config.aljazeera_feed_url)


def build_google_news_rss_url(config: PipelineConfig, query: str) -> str:
    """Construye una URL RSS de Google News para una query especifica."""

    return (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}"
        f"&hl={quote_plus(config.google_news_hl)}"
        f"&gl={quote_plus(config.google_news_gl)}"
        f"&ceid={quote_plus(config.google_news_ceid)}"
    )


def entry_publisher(entry: Any) -> str:
    """Extrae publisher de una entrada RSS si feedparser lo expone."""

    source = entry.get("source", {}) if hasattr(entry, "get") else {}
    if isinstance(source, dict):
        return strip_html(source.get("title", ""))
    return ""


def fetch_google_news_rss(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta Google News RSS con queries tematicas y normaliza resultados."""

    source_name = "google_news_rss"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    rows: list[dict[str, Any]] = []
    successful_feeds = 0
    for query in config.google_news_queries:
        rate_limiter.wait(source_name, config.google_news_min_interval_seconds)
        feed_url = build_google_news_rss_url(config, query)
        response = request_with_retries(
            session,
            "GET",
            feed_url,
            config=config,
            source_name=source_name,
            headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"},
        )
        if response is None:
            logging.warning("Google News RSS fallo para query: %s", query)
            continue

        successful_feeds += 1
        parsed = feedparser.parse(response.content)
        if getattr(parsed, "bozo", False):
            logging.warning(
                "%s RSS tiene advertencias de parsing para '%s': %s",
                source_name,
                query,
                parsed.bozo_exception,
            )

        for entry in getattr(parsed, "entries", []):
            title = strip_html(entry.get("title", ""))
            summary = strip_html(entry.get("summary", entry.get("description", "")))
            publisher = entry_publisher(entry)
            text_parts = [f"query={query}"]
            if publisher:
                text_parts.append(f"publisher={publisher}")
            if summary:
                text_parts.append(summary)
            text = "; ".join(text_parts)
            link = entry.get("link", "")
            published = entry.get("published") or entry.get("updated")
            country = infer_country_from_text(title, text)
            lat, lon = coords_for_country(country)
            rows.append(
                {
                    "timestamp": parse_any_timestamp(published),
                    "source": source_name,
                    "title": title,
                    "text": text,
                    "url": link,
                    "country": country or "",
                    "lat": lat,
                    "lon": lon,
                    "value": pd.NA,
                }
            )

    if successful_feeds == 0:
        logging.error("Google News RSS fallo en todas las queries; se continuara sin esta fuente.")
        return None

    df = ensure_normalized(pd.DataFrame(rows), source_name)
    if df is not None:
        df = df.drop_duplicates(subset=["title", "url"]).reset_index(drop=True)
        df = filter_thematic_news(df, source_name)
        save_csv(df, cache_path, source_name)
    return df


def fetch_opensky(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta estados actuales de aeronaves en OpenSky dentro del bbox configurado."""

    source_name = "opensky"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    rate_limiter.wait("opensky", 10.0)
    token_manager = OpenSkyTokenManager(
        session,
        config.opensky_client_id,
        config.opensky_client_secret,
        config,
    )
    lamin, lomin, lamax, lomax = config.opensky_bbox
    response = request_with_retries(
        session,
        "GET",
        "https://opensky-network.org/api/states/all",
        config=config,
        source_name=source_name,
        params={"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax},
        headers=token_manager.headers(),
    )
    if response is None:
        logging.error("OpenSky fallo; el pipeline continuara sin esta fuente.")
        return None

    payload = safe_json(response, source_name)
    if not isinstance(payload, dict):
        return None

    states = payload.get("states") or []
    if not isinstance(states, list):
        logging.warning("OpenSky JSON no contiene lista 'states'.")
        return None

    columns = [
        "icao24",
        "callsign",
        "origin_country",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "true_track",
        "vertical_rate",
        "sensors",
        "geo_altitude",
        "squawk",
        "spi",
        "position_source",
        "category",
    ]
    rows: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, list):
            continue
        values = state + [None] * (len(columns) - len(state))
        record = dict(zip(columns, values))
        callsign = strip_html(record.get("callsign") or "").strip()
        icao24 = record.get("icao24") or ""
        country = record.get("origin_country") or infer_country_from_coords(
            record.get("latitude"),
            record.get("longitude"),
        )
        title = f"Aircraft {callsign or icao24}".strip()
        text = (
            f"origin_country={country}; velocity_mps={record.get('velocity')}; "
            f"baro_altitude_m={record.get('baro_altitude')}; on_ground={record.get('on_ground')}"
        )
        rows.append(
            {
                "timestamp": parse_any_timestamp(record.get("last_contact") or payload.get("time")),
                "source": source_name,
                "title": title,
                "text": text,
                "url": "https://opensky-network.org/api/states/all",
                "country": country or "",
                "lat": record.get("latitude"),
                "lon": record.get("longitude"),
                "value": record.get("velocity"),
            }
        )

    df = ensure_normalized(pd.DataFrame(rows), source_name)
    if df is not None:
        save_csv(df, cache_path, source_name)
    return df


def build_nasa_firms_url(config: PipelineConfig, day_range: int | None = None) -> str | None:
    """Construye URL de NASA FIRMS sin hardcodear el MAP_KEY en el codigo."""

    if not config.nasa_firms_map_key:
        return None
    range_value = day_range if day_range is not None else config.nasa_firms_day_range
    parts = [
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
        config.nasa_firms_map_key,
        config.nasa_firms_source,
        config.nasa_firms_area,
        str(range_value),
    ]
    if config.nasa_firms_date:
        parts.append(config.nasa_firms_date)
    return "/".join(parts)


def sanitized_nasa_url(config: PipelineConfig, day_range: int | None = None) -> str:
    """URL descriptiva sin exponer el MAP_KEY en los CSV."""

    range_value = day_range if day_range is not None else config.nasa_firms_day_range
    parts = [
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
        "[MAP_KEY]",
        config.nasa_firms_source,
        config.nasa_firms_area,
        str(range_value),
    ]
    if config.nasa_firms_date:
        parts.append(config.nasa_firms_date)
    return "/".join(parts)


def parse_firms_timestamp(acq_date: Any, acq_time: Any) -> pd.Timestamp:
    """Convierte acq_date + acq_time de FIRMS a timestamp UTC."""

    if acq_date is None or pd.isna(acq_date):
        return pd.NaT
    time_text = str(acq_time or "0000").replace(".0", "").zfill(4)
    raw = f"{acq_date} {time_text[:2]}:{time_text[2:4]}"
    return pd.to_datetime(raw, utc=True, errors="coerce")


def empty_nasa_firms_frame() -> pd.DataFrame:
    """Crea un DataFrame FIRMS vacio pero con esquema completo."""

    columns = DEFAULT_COLUMNS + OPTIONAL_COLUMNS
    normalized = ensure_normalized(pd.DataFrame(columns=columns), "nasa_firms")
    return normalized if normalized is not None else pd.DataFrame(columns=columns)


def validate_nasa_firms_frame(df: pd.DataFrame) -> bool:
    """Valida campos criticos de FIRMS sin tumbar el pipeline."""

    required_columns = {"timestamp", "lat", "lon", "confidence", "satellite"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.warning("NASA FIRMS no tiene columnas requeridas: %s", sorted(missing_columns))
        return False

    if df.empty:
        logging.warning(
            "NASA FIRMS respondio correctamente, pero el CSV esta vacio para el bbox/dias configurados."
        )
        logging.info("[OK] NASA FIRMS esquema validado: lat/lon/timestamp/confidence/satellite")
        return True

    checks = {
        "latitudes": df["lat"].notna().any(),
        "longitudes": df["lon"].notna().any(),
        "timestamps": df["timestamp"].notna().any(),
        "confidence": df["confidence"].astype(str).str.strip().ne("").any(),
        "satellite": df["satellite"].astype(str).str.strip().ne("").any(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        logging.warning("NASA FIRMS validacion incompleta. Campos sin datos: %s", failed)
        return False

    logging.info("[OK] NASA FIRMS validado: lat/lon/timestamp/confidence/satellite")
    return True


def nasa_firms_day_ranges_to_try(config: PipelineConfig) -> list[int]:
    """Devuelve rangos de dias para FIRMS: principal y fallback regional opcional."""

    primary = max(int(config.nasa_firms_day_range), 1)
    ranges = [primary]
    fallback = config.nasa_firms_empty_retry_day_range
    if fallback is not None and fallback > primary:
        ranges.append(fallback)
    return ranges


def nasa_firms_day_range_from_error(message: str) -> int | None:
    """Extrae el maximo de dias permitido por FIRMS cuando la API lo informa."""

    match = re.search(r"\[(\d+)\.\.(\d+)\]", message)
    if not match:
        return None
    return int(match.group(2))


def normalize_firms_rows(
    raw: pd.DataFrame,
    config: PipelineConfig,
    source_name: str,
    day_range: int,
) -> pd.DataFrame | None:
    """Transforma el CSV crudo de FIRMS al esquema ML normalizado."""

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        country = infer_country_from_coords(lat, lon)
        confidence = row.get("confidence", "")
        satellite = row.get("satellite", "")
        instrument = row.get("instrument", "")
        frp = row.get("frp", pd.NA)
        bright = row.get("bright_ti4", row.get("brightness", pd.NA))
        rows.append(
            {
                "timestamp": parse_firms_timestamp(row.get("acq_date"), row.get("acq_time")),
                "source": source_name,
                "title": (
                    f"FIRMS hotspot {satellite} {instrument} confidence={confidence}"
                ).strip(),
                "text": (
                    f"satellite={satellite}; instrument={instrument}; "
                    f"confidence={confidence}; frp={frp}; brightness={bright}; "
                    f"daynight={row.get('daynight', '')}"
                ),
                "url": sanitized_nasa_url(config, day_range),
                "country": country or "",
                "lat": lat,
                "lon": lon,
                "value": frp if not pd.isna(frp) else bright,
                "confidence": confidence,
                "satellite": satellite,
            }
        )
    return ensure_normalized(pd.DataFrame(rows), source_name)


def fetch_nasa_firms(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Descarga detecciones FIRMS regionales en CSV usando NASA_FIRMS_MAP_KEY."""

    source_name = "nasa_firms"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        cached = load_cached_csv(cache_path, source_name)
        if cached is not None and {"confidence", "satellite"}.issubset(cached.columns):
            return cached
        logging.info("NASA FIRMS cache sin columnas nuevas; se consultara la API.")

    if not config.nasa_firms_map_key:
        logging.warning(
            "NASA FIRMS desactivado funcionalmente: falta NASA_FIRMS_MAP_KEY en .env."
        )
        return None

    day_ranges = nasa_firms_day_ranges_to_try(config)
    attempted_day_ranges: set[int] = set()
    while day_ranges:
        day_range = day_ranges.pop(0)
        if day_range in attempted_day_ranges:
            continue
        attempted_day_ranges.add(day_range)
        url = build_nasa_firms_url(config, day_range)
        if not url:
            return None

        rate_limiter.wait("nasa_firms", 5.0)
        logging.info(
            "NASA FIRMS solicitando Area API regional: source=%s bbox=%s days=%s",
            config.nasa_firms_source,
            config.nasa_firms_area,
            day_range,
        )
        response = request_with_retries(
            session,
            "GET",
            url,
            config=config,
            source_name=source_name,
            return_response_on_statuses={400},
            headers={"Accept": "text/csv, */*"},
        )
        if response is None:
            logging.error("NASA FIRMS fallo; el pipeline continuara sin esta fuente.")
            return None
        if response.status_code == 400 and "Invalid day range" in response.text:
            compatible_day_range = nasa_firms_day_range_from_error(response.text)
            if compatible_day_range and compatible_day_range not in attempted_day_ranges:
                logging.warning(
                    "NASA FIRMS rechazo %s dias (%s). Se probara %s dias en el mismo bbox regional.",
                    day_range,
                    response.text[:120],
                    compatible_day_range,
                )
                day_ranges = [
                    queued_range
                    for queued_range in day_ranges
                    if queued_range <= compatible_day_range
                ]
                day_ranges.insert(0, compatible_day_range)
                continue
            logging.warning("NASA FIRMS rechazo day_range y no hay fallback valido: %s", response.text)
            return None

        logging.info("[OK] NASA FIRMS conectado")
        text = response.text.strip()
        if not text or "Invalid" in text[:200] or (
            "MAP_KEY" in text[:200] and "," not in text[:200]
        ):
            logging.warning("NASA FIRMS devolvio una respuesta no tabular: %s", text[:200])
            return None
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if "," not in first_line:
            logging.warning("NASA FIRMS no devolvio encabezado CSV valido: %s", first_line[:200])
            return None

        try:
            raw = pd.read_csv(io.StringIO(response.text))
        except pd.errors.ParserError as exc:
            logging.warning("NASA FIRMS CSV invalido: %s", exc)
            return None

        required = {"latitude", "longitude", "acq_date", "acq_time"}
        if not required.issubset(set(raw.columns)):
            logging.warning(
                "NASA FIRMS no trae columnas esperadas. Columnas: %s",
                list(raw.columns),
            )
            return None

        for metadata_column in ["confidence", "satellite"]:
            if metadata_column not in raw.columns:
                logging.warning("NASA FIRMS no trae columna %s; se creara vacia.", metadata_column)
                raw[metadata_column] = pd.NA

        if raw.empty:
            df = empty_nasa_firms_frame()
            validate_nasa_firms_frame(df)
            if day_ranges:
                logging.info(
                    "NASA FIRMS vacio con %s dias; se probara %s dias en el mismo bbox regional.",
                    day_range,
                    day_ranges[0],
                )
                continue
            if save_csv(df, cache_path, source_name):
                logging.info("[OK] nasa_firms.csv generado")
            return df

        df = normalize_firms_rows(raw, config, source_name, day_range)
        if df is not None:
            validate_nasa_firms_frame(df)
            if save_csv(df, cache_path, source_name):
                logging.info("[OK] nasa_firms.csv generado")
        return df

    return empty_nasa_firms_frame()


def normalize_acled_rows(payload: dict[str, Any] | list[Any], source_name: str) -> pd.DataFrame | None:
    """Transforma respuesta ACLED al esquema comun y conserva campos de evento."""

    if isinstance(payload, dict):
        records = payload.get("data") or payload.get("results") or payload.get("events") or []
    else:
        records = payload
    if not isinstance(records, list):
        logging.warning("ACLED no devolvio una lista de eventos reconocible.")
        return None

    rows: list[dict[str, Any]] = []
    for event in records:
        if not isinstance(event, dict):
            continue
        actor1 = strip_html(event.get("actor1", ""))
        actor2 = strip_html(event.get("actor2", ""))
        event_type = strip_html(event.get("event_type", ""))
        sub_event_type = strip_html(event.get("sub_event_type", ""))
        notes = strip_html(event.get("notes", ""))
        country = strip_html(event.get("country", ""))
        title = " | ".join(part for part in [event_type, country, actor1] if part)
        text = (
            f"actor1={actor1}; actor2={actor2}; event_type={event_type}; "
            f"sub_event_type={sub_event_type}; notes={notes}"
        )
        rows.append(
            {
                "timestamp": parse_any_timestamp(event.get("event_date") or event.get("timestamp")),
                "source": source_name,
                "title": title or "ACLED event",
                "text": text,
                "url": str(event.get("source") or "https://acleddata.com/"),
                "country": country,
                "lat": event.get("latitude"),
                "lon": event.get("longitude"),
                "value": event.get("fatalities"),
                "actor1": actor1,
                "actor2": actor2,
                "event_date": event.get("event_date"),
                "fatalities": event.get("fatalities"),
                "event_type": event_type,
                "sub_event_type": sub_event_type,
                "notes": notes,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keywords = [keyword.lower() for keyword in DEFAULT_ACLED_KEYWORDS]
    haystack = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
        + " "
        + df["country"].fillna("").astype(str)
    ).str.lower()
    mask = haystack.apply(lambda value: any(keyword.lower() in value for keyword in keywords))
    filtered = df[mask].reset_index(drop=True)
    return filtered if not filtered.empty else df


def fetch_acled(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta ACLED API para eventos regionales si hay credenciales."""

    source_name = "acled"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    token_headers = ACLEDTokenManager(session, config).headers()
    if token_headers is None:
        logging.warning(
            "ACLED omitido: define ACLED_API_KEY o ACLED_USERNAME/ACLED_PASSWORD en .env."
        )
        return None

    rate_limiter.wait(source_name, 10.0)
    fields = "|".join(
        [
            "event_date",
            "actor1",
            "actor2",
            "country",
            "fatalities",
            "latitude",
            "longitude",
            "event_type",
            "sub_event_type",
            "notes",
            "source",
        ]
    )
    params = {
        "_format": "json",
        "limit": max(1, min(config.acled_limit, 5000)),
        "fields": fields,
        "country": acled_country_filter(config.acled_countries),
        "event_date": f"{ymd_days_ago(config.acled_days)}|{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "event_date_where": "BETWEEN",
    }
    response = request_with_retries(
        session,
        "GET",
        "https://acleddata.com/api/acled/read",
        config=config,
        source_name=source_name,
        params=params,
        headers=token_headers,
    )
    if response is None:
        logging.error("ACLED fallo; el pipeline continuara sin esta fuente.")
        return None

    payload = safe_json(response, source_name)
    if payload is None:
        return None
    df = normalize_acled_rows(payload, source_name)
    if df is not None:
        save_csv(df, cache_path, source_name)
    return df


def fetch_worldpop(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta metadatos WorldPop por pais para exposicion poblacional."""

    source_name = "worldpop"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    rows: list[dict[str, Any]] = []
    for iso3 in config.worldpop_countries:
        iso3 = iso3.strip().upper()
        if not iso3:
            continue
        rate_limiter.wait(source_name, config.worldpop_min_interval_seconds)
        response = request_with_retries(
            session,
            "GET",
            f"https://www.worldpop.org/rest/data/pop/{config.worldpop_dataset}",
            config=config,
            source_name=source_name,
            params={"iso3": iso3},
        )
        if response is None:
            continue
        payload = safe_json(response, source_name)
        if not isinstance(payload, dict):
            continue
        records = payload.get("data") or payload.get("files") or []
        if isinstance(records, dict):
            records = [records]
        country = iso3_to_country_name(iso3)
        lat, lon = coords_for_country(country)
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            record_text = json.dumps(record, ensure_ascii=True)
            record_year = str(record.get("popyear") or record.get("year") or "")
            if record_year and record_year != str(config.worldpop_year):
                continue
            if not record_year and str(config.worldpop_year) not in record_text:
                continue
            title = strip_html(
                record.get("title")
                or record.get("name")
                or f"WorldPop {config.worldpop_dataset} {iso3} {config.worldpop_year}"
            )
            rows.append(
                {
                    "timestamp": parse_any_timestamp(f"{config.worldpop_year}-01-01"),
                    "source": source_name,
                    "title": title,
                    "text": record_text,
                    "url": record.get("url") or record.get("download_url") or "https://www.worldpop.org",
                    "country": country,
                    "lat": lat,
                    "lon": lon,
                    "value": record.get("value") or record.get("population"),
                    "dataset": config.worldpop_dataset,
                    "iso3": iso3,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        save_csv(df, cache_path, source_name)
    return df


def normalize_unhcr_rows(
    payload: Any,
    source_name: str,
    country_filter: set[str],
) -> pd.DataFrame | None:
    """Normaliza respuestas UNHCR Refugee Data Finder de forma tolerante."""

    if isinstance(payload, dict):
        records = payload.get("items") or payload.get("data") or payload.get("results") or []
    elif isinstance(payload, list):
        records = payload
    else:
        return None
    if not isinstance(records, list):
        return None

    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        coo = str(record.get("coo") or record.get("coo_iso") or record.get("origin") or "").upper()
        coa = str(record.get("coa") or record.get("coa_iso") or record.get("asylum") or "").upper()
        if country_filter and not ({coo, coa} & country_filter):
            continue
        country = iso3_to_country_name(coa or coo)
        lat, lon = coords_for_country(country)
        year = record.get("year") or record.get("date") or datetime.now(timezone.utc).year
        refugees = parse_int(record.get("refugees") or record.get("refugee") or record.get("total"), 0)
        asylum_seekers = parse_int(record.get("asylum_seekers") or record.get("asylum"), 0)
        idps = parse_int(record.get("idps") or record.get("idp"), 0)
        value = refugees + asylum_seekers + idps
        rows.append(
            {
                "timestamp": parse_any_timestamp(f"{year}-01-01"),
                "source": source_name,
                "title": f"UNHCR displacement data {coo or 'origin'} to {coa or 'asylum'}",
                "text": json.dumps(record, ensure_ascii=True),
                "url": "https://www.unhcr.org/refugee-statistics/",
                "country": country,
                "lat": lat,
                "lon": lon,
                "value": value if value > 0 else pd.NA,
                "coo": coo,
                "coa": coa,
                "year": year,
                "refugees": refugees,
                "asylum_seekers": asylum_seekers,
                "idps": idps,
            }
        )
    return pd.DataFrame(rows)


def fetch_unhcr(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta UNHCR Refugee Data Finder API para desplazamiento regional."""

    source_name = "unhcr"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    countries = ",".join(country.strip().upper() for country in config.unhcr_countries if country.strip())
    frames: list[pd.DataFrame] = []
    for filter_name in ["coo", "coa"]:
        rate_limiter.wait(source_name, 2.0)
        response = request_with_retries(
            session,
            "GET",
            "https://api.unhcr.org/population/v1/population/",
            config=config,
            source_name=source_name,
            params={
                "year": config.unhcr_year,
                "limit": 1000,
                "cf_type": "ISO",
                filter_name: countries,
            },
        )
        if response is None:
            continue
        payload = safe_json(response, source_name)
        partial = normalize_unhcr_rows(
            payload,
            source_name,
            {country.strip().upper() for country in config.unhcr_countries},
        )
        if partial is not None and not partial.empty:
            frames.append(partial)
    if not frames:
        logging.warning("UNHCR no devolvio datos regionales para el filtro configurado.")
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["timestamp", "title", "country", "coo", "coa"]
    )
    if df is not None and not df.empty:
        save_csv(df, cache_path, source_name)
    return df


def fetch_hdx(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Busca datasets humanitarios en HDX via CKAN package_search."""

    source_name = "hdx"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    rows: list[dict[str, Any]] = []
    for query in config.hdx_queries:
        rate_limiter.wait(source_name, config.hdx_min_interval_seconds)
        response = request_with_retries(
            session,
            "GET",
            "https://data.humdata.org/api/3/action/package_search",
            config=config,
            source_name=source_name,
            params={"q": query, "rows": max(1, min(config.hdx_rows, 100))},
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
            },
        )
        if response is None:
            continue
        payload = safe_json(response, source_name)
        if not isinstance(payload, dict) or not payload.get("success", False):
            continue
        result = payload.get("result", {})
        packages = result.get("results", []) if isinstance(result, dict) else []
        for package in packages:
            if not isinstance(package, dict):
                continue
            title = strip_html(package.get("title") or package.get("name") or "HDX dataset")
            notes = strip_html(package.get("notes") or "")
            country = infer_country_from_text(title, notes) or ""
            lat, lon = coords_for_country(country)
            rows.append(
                {
                    "timestamp": parse_any_timestamp(package.get("metadata_modified") or package.get("metadata_created")),
                    "source": source_name,
                    "title": title,
                    "text": notes,
                    "url": f"https://data.humdata.org/dataset/{package.get('name', '')}",
                    "country": country,
                    "lat": lat,
                    "lon": lon,
                    "value": package.get("num_resources"),
                    "query": query,
                    "organization": (package.get("organization") or {}).get("title", ""),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        save_csv(df, cache_path, source_name)
    return df


def get_sentinel_hub_token(session: Session, config: PipelineConfig) -> str | None:
    """Obtiene token OAuth2 de Sentinel Hub con client credentials."""

    if not (config.sentinel_hub_client_id and config.sentinel_hub_client_secret):
        logging.warning(
            "Sentinel Hub omitido: define SENTINEL_HUB_CLIENT_ID y SENTINEL_HUB_CLIENT_SECRET."
        )
        return None
    response = request_with_retries(
        session,
        "POST",
        "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token",
        config=config,
        source_name="sentinel_hub_auth",
        data={
            "grant_type": "client_credentials",
            "client_id": config.sentinel_hub_client_id,
            "client_secret": config.sentinel_hub_client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if response is None:
        return None
    payload = safe_json(response, "sentinel_hub_auth")
    if not isinstance(payload, dict):
        return None
    return payload.get("access_token")


def fetch_sentinel_hub(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta Sentinel Hub Catalog API para disponibilidad satelital regional."""

    source_name = "sentinel_hub"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    token = get_sentinel_hub_token(session, config)
    if not token:
        return None

    rate_limiter.wait(source_name, 2.0)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(config.sentinel_hub_days, 1))
    bbox = [float(part.strip()) for part in config.sentinel_hub_bbox.split(",")]
    response = request_with_retries(
        session,
        "POST",
        "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search",
        config=config,
        source_name=source_name,
        json={
            "collections": [config.sentinel_hub_collection],
            "bbox": bbox,
            "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}",
            "limit": max(1, min(config.sentinel_hub_limit, 100)),
        },
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    if response is None:
        return None
    payload = safe_json(response, source_name)
    if not isinstance(payload, dict):
        return None
    center_lat, center_lon = bbox_center_from_lonlat(config.sentinel_hub_bbox)
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        scene_id = feature.get("id") or props.get("id") or "Sentinel scene"
        cloud = props.get("eo:cloud_cover") or props.get("cloudCover")
        rows.append(
            {
                "timestamp": parse_any_timestamp(props.get("datetime")),
                "source": source_name,
                "title": f"{config.sentinel_hub_collection} scene {scene_id}",
                "text": json.dumps(props, ensure_ascii=True),
                "url": "https://www.sentinel-hub.com/",
                "country": "regional_bbox",
                "lat": center_lat,
                "lon": center_lon,
                "value": cloud,
                "collection": config.sentinel_hub_collection,
                "scene_id": scene_id,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        save_csv(df, cache_path, source_name)
    return df


def fetch_openstreetmap(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta Overpass/OpenStreetMap para infraestructura y conectividad."""

    source_name = "openstreetmap"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    rows: list[dict[str, Any]] = []
    overpass_bbox = lonlat_bbox_to_overpass(config.osm_bbox)
    lat_min, lon_min, lat_max, lon_max = [float(part.strip()) for part in config.osm_bbox.split(",")]
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    for label, tag_filter in config.osm_tags.items():
        rate_limiter.wait(source_name, config.osm_min_interval_seconds)
        query = f"""
        [out:json][timeout:60];
        (
          node{tag_filter}({overpass_bbox});
          way{tag_filter}({overpass_bbox});
          relation{tag_filter}({overpass_bbox});
        );
        out count;
        """
        payload = None
        for overpass_url in config.osm_overpass_urls:
            response = request_with_retries(
                session,
                "GET",
                overpass_url,
                config=config,
                source_name=source_name,
                params={"data": query},
                headers={"Accept": "*/*"},
            )
            if response is None:
                continue
            payload = safe_json(response, source_name)
            if isinstance(payload, dict):
                break
        if not isinstance(payload, dict):
            continue
        elements = payload.get("elements", [])
        if isinstance(elements, list) and elements and isinstance(elements[0], dict):
            count = parse_int((elements[0].get("tags") or {}).get("total"), 0)
        else:
            count = 0
        rows.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "source": source_name,
                "title": f"OSM {label} features in regional bbox",
                "text": f"OpenStreetMap/Overpass count for {tag_filter} in bbox {config.osm_bbox}.",
                "url": "https://www.openstreetmap.org",
                "country": "regional_bbox",
                "lat": center_lat,
                "lon": center_lon,
                "value": count,
                "feature_type": label,
                "tag_filter": tag_filter,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        save_csv(df, cache_path, source_name)
    return df


def fetch_google_earth_engine(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Consulta Google Earth Engine si la libreria y autenticacion local existen."""

    del session, rate_limiter
    source_name = "google_earth_engine"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    try:
        import ee
    except ImportError:
        logging.warning(
            "Google Earth Engine omitido: instala earthengine-api y ejecuta earthengine authenticate."
        )
        return None

    try:
        if config.gee_project:
            ee.Initialize(project=config.gee_project)
        else:
            ee.Initialize()
        west, south, east, north = [float(part.strip()) for part in config.gee_bbox.split(",")]
        region = ee.Geometry.Rectangle([west, south, east, north])
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(config.gee_days, 1))
        collection = (
            ee.ImageCollection(config.gee_collection)
            .filterBounds(region)
            .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        )
        image_count = collection.size().getInfo()
    except Exception as exc:
        logging.warning("Google Earth Engine no pudo consultar la coleccion: %s", exc)
        return None

    center_lat, center_lon = bbox_center_from_lonlat(config.gee_bbox)
    df = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "source": source_name,
                "title": f"GEE image count for {config.gee_collection}",
                "text": f"Earth Engine collection count over bbox {config.gee_bbox} for last {config.gee_days} days.",
                "url": "https://earthengine.google.com/",
                "country": "regional_bbox",
                "lat": center_lat,
                "lon": center_lon,
                "value": image_count,
                "collection": config.gee_collection,
            }
        ]
    )
    save_csv(df, cache_path, source_name)
    return df


def discover_ukmto_links(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> list[tuple[str, str]]:
    """Descubre PDFs o paginas de reportes UKMTO desde paginas publicas."""

    discovered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for page_url in config.ukmto_pages:
        rate_limiter.wait("ukmto", 3.0)
        response = request_with_retries(
            session,
            "GET",
            page_url,
            config=config,
            source_name="ukmto",
            headers={"Accept": "text/html,application/xhtml+xml,application/pdf,*/*"},
        )
        if response is None:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = str(link.get("href", ""))
            text = strip_html(link.get_text(" ", strip=True))
            absolute = urljoin(page_url, href)
            lower = absolute.lower()
            if lower.startswith(("mailto:", "tel:")):
                continue
            is_report_link = (
                ".pdf" in lower
                or "warning" in lower
                or "advisory" in lower
                or "recent-incidents#" in lower
                or "/recent-incidents/" in lower
            )
            if not is_report_link:
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            discovered.append((absolute, text))
            if len(discovered) >= config.ukmto_max_reports:
                return discovered
    return discovered


def ukmto_report_text(
    session: Session,
    config: PipelineConfig,
    url: str,
    link_text: str,
) -> str:
    """Descarga un reporte UKMTO y extrae texto tolerando HTML/PDF."""

    response = request_with_retries(
        session,
        "GET",
        url,
        config=config,
        source_name="ukmto",
        headers={"Accept": "application/pdf,text/html,*/*"},
    )
    if response is None:
        return link_text
    content_type = response.headers.get("Content-Type", "").lower()
    if url.lower().endswith(".pdf") or "pdf" in content_type:
        pdf_text = parse_pdf_text(response.content, "ukmto")
        return pdf_text or link_text or url
    return BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)


def fetch_ukmto(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Scrapea reportes publicos UKMTO y normaliza incidentes maritimos."""

    source_name = "ukmto"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    links = discover_ukmto_links(session, config, rate_limiter)
    if not links:
        logging.warning("UKMTO no expuso reportes parseables; se generara CSV vacio con esquema estable.")
        empty = pd.DataFrame(
            columns=DEFAULT_COLUMNS + ["location", "incident_type", "description"]
        )
        save_csv(empty, cache_path, source_name)
        return empty

    rows: list[dict[str, Any]] = []
    for url, link_text in links[: config.ukmto_max_reports]:
        text = ukmto_report_text(session, config, url, link_text)
        if not text:
            continue
        timestamp = extract_ukmto_timestamp(text, url)
        incident_type = detect_incident_type(text)
        location = extract_ukmto_location(text)
        lat, lon = parse_coordinate_pair(text)
        country = infer_country_from_text(location, text) or infer_country_from_coords(lat, lon) or ""
        title = strip_html(link_text) or f"UKMTO {incident_type.replace('_', ' ')}".title()
        if len(title) < 8:
            title = f"UKMTO {incident_type.replace('_', ' ')}"
        rows.append(
            {
                "timestamp": timestamp,
                "source": source_name,
                "title": title,
                "text": re.sub(r"\s+", " ", text).strip()[:2500],
                "url": url,
                "country": country,
                "lat": lat,
                "lon": lon,
                "value": pd.NA,
                "location": location,
                "incident_type": incident_type,
                "description": re.sub(r"\s+", " ", text).strip()[:2500],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["url", "timestamp", "title"]).reset_index(drop=True)
        save_csv(df, cache_path, source_name)
    return df


def bluesky_auth_headers(session: Session, config: PipelineConfig) -> tuple[str, dict[str, str]]:
    """Devuelve host y headers para Bluesky; auth es opcional."""

    if not (config.bluesky_handle and config.bluesky_app_password):
        return "https://api.bsky.app", {}

    response = request_with_retries(
        session,
        "POST",
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        config=config,
        source_name="bluesky_auth",
        json={
            "identifier": config.bluesky_handle,
            "password": config.bluesky_app_password,
        },
        headers={"Content-Type": "application/json"},
    )
    if response is None:
        logging.warning("Bluesky auth fallo; se usara AppView publico.")
        return "https://api.bsky.app", {}
    payload = safe_json(response, "bluesky_auth")
    if not isinstance(payload, dict) or "accessJwt" not in payload:
        logging.warning("Bluesky auth invalida; se usara AppView publico.")
        return "https://api.bsky.app", {}
    return "https://bsky.social", {"Authorization": f"Bearer {payload['accessJwt']}"}


def fetch_bluesky(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Busca posts Bluesky relacionados con el conflicto usando AT Protocol AppView."""

    source_name = "bluesky"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)

    host, headers = bluesky_auth_headers(session, config)
    rows: list[dict[str, Any]] = []
    for query in config.bluesky_queries:
        rate_limiter.wait(source_name, 2.0)
        response = request_with_retries(
            session,
            "GET",
            f"{host}/xrpc/app.bsky.feed.searchPosts",
            config=config,
            source_name=source_name,
            params={
                "q": query,
                "limit": max(1, min(config.bluesky_limit, 100)),
                "sort": "latest",
                "since": iso_utc_days_ago(config.bluesky_days),
            },
            headers=headers,
        )
        if response is None:
            logging.warning("Bluesky fallo para query: %s", query)
            continue
        payload = safe_json(response, source_name)
        if not isinstance(payload, dict):
            continue
        for post in payload.get("posts", []):
            if not isinstance(post, dict):
                continue
            author = post.get("author", {}) if isinstance(post.get("author"), dict) else {}
            record = post.get("record", {}) if isinstance(post.get("record"), dict) else {}
            text = strip_html(record.get("text", ""))
            handle = strip_html(author.get("handle", ""))
            uri = str(post.get("uri", ""))
            likes = parse_int(post.get("likeCount"))
            reposts = parse_int(post.get("repostCount"))
            url = bluesky_post_url(uri, handle)
            country = infer_country_from_text("", text)
            lat, lon = coords_for_country(country)
            rows.append(
                {
                    "timestamp": parse_any_timestamp(record.get("createdAt") or post.get("indexedAt")),
                    "source": source_name,
                    "title": text[:120] or f"Bluesky post {handle}",
                    "text": text,
                    "url": url,
                    "country": country or "",
                    "lat": lat,
                    "lon": lon,
                    "value": likes + reposts,
                    "author": handle,
                    "likes": likes,
                    "reposts": reposts,
                    "uri": uri,
                    "query": query,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["uri", "url", "text"]).reset_index(drop=True)
        df = filter_thematic_news(df, source_name)
        save_csv(df, cache_path, source_name)
    return df


def fetch_youtube_video_details(
    session: Session,
    config: PipelineConfig,
    video_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Consulta videos.list para estadisticas y metadata persistente."""

    details: dict[str, dict[str, Any]] = {}
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start : start + 50]
        if not chunk:
            continue
        response = request_with_retries(
            session,
            "GET",
            "https://www.googleapis.com/youtube/v3/videos",
            config=config,
            source_name="youtube",
            params={
                "part": "snippet,statistics",
                "id": ",".join(chunk),
                "key": config.youtube_api_key,
            },
        )
        if response is None:
            continue
        payload = safe_json(response, "youtube")
        if not isinstance(payload, dict):
            continue
        for item in payload.get("items", []):
            if isinstance(item, dict) and item.get("id"):
                details[str(item["id"])] = item
    return details


def fetch_youtube(
    session: Session,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
) -> pd.DataFrame | None:
    """Busca metadata de videos YouTube con Data API v3."""

    source_name = "youtube"
    cache_path = config.data_dir / SOURCE_OUTPUTS[source_name]
    if should_use_cache(cache_path, config):
        return load_cached_csv(cache_path, source_name)
    if not config.youtube_api_key:
        logging.warning("YouTube omitido: falta YOUTUBE_API_KEY en .env.")
        return None

    search_items: dict[str, dict[str, Any]] = {}
    for query in config.youtube_queries:
        rate_limiter.wait(source_name, 1.5)
        response = request_with_retries(
            session,
            "GET",
            "https://www.googleapis.com/youtube/v3/search",
            config=config,
            source_name=source_name,
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "order": "date",
                "maxResults": max(1, min(config.youtube_max_results, 50)),
                "publishedAfter": iso_utc_days_ago(config.youtube_days),
                "key": config.youtube_api_key,
            },
        )
        if response is None:
            logging.warning("YouTube fallo para query: %s", query)
            continue
        payload = safe_json(response, source_name)
        if not isinstance(payload, dict):
            continue
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            video_id = item.get("id", {}).get("videoId") if isinstance(item.get("id"), dict) else None
            if video_id:
                item["query"] = query
                search_items[str(video_id)] = item

    if not search_items:
        return None

    details = fetch_youtube_video_details(session, config, list(search_items))
    rows: list[dict[str, Any]] = []
    for video_id, item in search_items.items():
        detail = details.get(video_id, item)
        snippet = detail.get("snippet", {}) if isinstance(detail.get("snippet"), dict) else {}
        statistics = detail.get("statistics", {}) if isinstance(detail.get("statistics"), dict) else {}
        title = strip_html(snippet.get("title", ""))
        description = strip_html(snippet.get("description", ""))
        channel = strip_html(snippet.get("channelTitle", ""))
        views = parse_int(statistics.get("viewCount"))
        likes = parse_int(statistics.get("likeCount"))
        country = infer_country_from_text(title, description)
        lat, lon = coords_for_country(country)
        rows.append(
            {
                "timestamp": parse_any_timestamp(snippet.get("publishedAt")),
                "source": source_name,
                "title": title,
                "text": description,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "country": country or "",
                "lat": lat,
                "lon": lon,
                "value": views,
                "channel": channel,
                "publish_date": snippet.get("publishedAt"),
                "views": views,
                "likes": likes,
                "video_id": video_id,
                "query": item.get("query", ""),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = filter_thematic_news(df, source_name)
        save_csv(df, cache_path, source_name)
    return df


FETCHERS: dict[str, Callable[[Session, PipelineConfig, RateLimiter], pd.DataFrame | None]] = {
    "gdelt": fetch_gdelt,
    "bbc_rss": fetch_bbc_rss,
    "aljazeera_rss": fetch_aljazeera_rss,
    "google_news_rss": fetch_google_news_rss,
    "opensky": fetch_opensky,
    "nasa_firms": fetch_nasa_firms,
    "acled": fetch_acled,
    "worldpop": fetch_worldpop,
    "unhcr": fetch_unhcr,
    "hdx": fetch_hdx,
    "sentinel_hub": fetch_sentinel_hub,
    "openstreetmap": fetch_openstreetmap,
    "google_earth_engine": fetch_google_earth_engine,
    "ukmto": fetch_ukmto,
    "bluesky": fetch_bluesky,
    "youtube": fetch_youtube,
}


def integrate_datasets(frames: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Une las fuentes disponibles y conserva un esquema comun."""

    valid_frames: list[pd.DataFrame] = []
    for source_name, df in frames.items():
        if source_name == "integrated" or df is None:
            continue
        normalized = ensure_normalized(df, source_name)
        if normalized is None:
            continue
        valid_frames.append(filter_thematic_news(normalized, source_name))
    if not valid_frames:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    integrated = pd.concat(valid_frames, ignore_index=True)
    integrated = integrated.drop_duplicates(
        subset=["source", "timestamp", "title", "url", "lat", "lon"]
    )
    integrated = integrated.sort_values(
        by=["timestamp", "source"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    normalized = ensure_normalized(integrated, "integrated")
    if normalized is None:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    return normalized


def log_source_summary(
    results: dict[str, pd.DataFrame | None],
    integrated: pd.DataFrame,
) -> None:
    """Muestra resumen por fuente antes y despues de deduplicar."""

    integrated_counts = (
        integrated["source"].value_counts().to_dict() if "source" in integrated.columns else {}
    )
    logging.info("Resumen por fuente:")
    for source_name in FETCHERS:
        frame = results.get(source_name)
        available_rows = len(frame) if frame is not None else 0
        integrated_rows = int(integrated_counts.get(source_name, 0))
        logging.info(
            "- %s: %s filas disponibles; %s filas en dataset_integrado.csv",
            source_name,
            available_rows,
            integrated_rows,
        )


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, pd.DataFrame | None]:
    """
    Ejecuta todas las fuentes habilitadas.

    Retorna un diccionario con DataFrames por fuente y la llave 'integrated'.
    """

    config = config or PipelineConfig.from_env()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    session = build_session(config)
    rate_limiter = RateLimiter()
    results: dict[str, pd.DataFrame | None] = {}

    logging.info("Iniciando pipeline OSINT. Carpeta data: %s", config.data_dir)
    for source_name, fetcher in FETCHERS.items():
        if not config.enabled_sources.get(source_name, False):
            logging.info("%s deshabilitado por configuracion.", source_name)
            results[source_name] = None
            continue
        logging.info("Ejecutando fuente: %s", source_name)
        try:
            results[source_name] = fetcher(session, config, rate_limiter)
        except Exception:
            logging.exception("%s fallo con excepcion inesperada; se continua.", source_name)
            results[source_name] = None
        if results[source_name] is None:
            csv_path = config.data_dir / SOURCE_OUTPUTS[source_name]
            results[source_name] = load_available_source_csv(csv_path, source_name)

    integrated = integrate_datasets(results)
    integrated_path = config.data_dir / "dataset_integrado.csv"
    save_csv(integrated, integrated_path, "dataset_integrado")
    results["integrated"] = integrated
    log_source_summary(results, integrated)
    logging.info("Pipeline finalizado. Filas integradas: %s", len(integrated))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumentos CLI para ejecutar el pipeline como script."""

    parser = argparse.ArgumentParser(description="Pipeline OSINT Iran-Israel-EE.UU. para ML1")
    parser.add_argument("--data-dir", default=None, help="Carpeta de salida CSV. Default: data/")
    parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Fuentes a activar separadas por coma. Ej: "
            "gdelt,bbc_rss,google_news_rss,acled,worldpop,unhcr,hdx,openstreetmap"
        ),
    )
    parser.add_argument("--disable-cache", action="store_true", help="Ignora CSV cacheados.")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--gdelt-query", default=None, help="Query GDELT DOC 2.0")
    parser.add_argument("--gdelt-timespan", default=None, help="Ej: 6h, 24h, 7d, 1week")
    parser.add_argument("--gdelt-maxrecords", type=int, default=None, help="1 a 250")
    parser.add_argument(
        "--google-news-queries",
        default=None,
        help="Queries Google News separadas por punto y coma.",
    )
    parser.add_argument(
        "--opensky-bbox",
        default=None,
        help="lat_min,lon_min,lat_max,lon_max. Ej: 24,34,40,64",
    )
    parser.add_argument(
        "--nasa-area",
        default=None,
        help="west,south,east,north. Ej: 20,10,80,50",
    )
    parser.add_argument(
        "--nasa-day-range",
        type=int,
        default=None,
        help="Dias FIRMS a consultar. Default regional: 7.",
    )
    parser.add_argument(
        "--nasa-empty-retry-day-range",
        type=int,
        default=None,
        help="Fallback opcional si FIRMS devuelve 0 filas. Ej: 14.",
    )
    parser.add_argument("--acled-days", type=int, default=None, help="Ventana ACLED en dias.")
    parser.add_argument("--acled-limit", type=int, default=None, help="Limite de eventos ACLED.")
    parser.add_argument("--worldpop-year", type=int, default=None, help="Anio WorldPop.")
    parser.add_argument(
        "--worldpop-countries",
        default=None,
        help="ISO3 WorldPop separados por punto y coma. Ej: IRN;ISR;PSE",
    )
    parser.add_argument("--unhcr-year", type=int, default=None, help="Anio UNHCR.")
    parser.add_argument(
        "--unhcr-countries",
        default=None,
        help="ISO3 UNHCR separados por punto y coma. Ej: IRN;ISR;PSE",
    )
    parser.add_argument(
        "--hdx-queries",
        default=None,
        help="Queries HDX separadas por punto y coma.",
    )
    parser.add_argument(
        "--sentinel-hub-bbox",
        default=None,
        help="west,south,east,north. Ej: 34,24,64,40",
    )
    parser.add_argument("--sentinel-hub-days", type=int, default=None, help="Ventana Sentinel Hub en dias.")
    parser.add_argument(
        "--osm-bbox",
        default=None,
        help="lat_min,lon_min,lat_max,lon_max. Ej: 24,34,40,64",
    )
    parser.add_argument(
        "--gee-bbox",
        default=None,
        help="west,south,east,north para Google Earth Engine. Ej: 34,24,64,40",
    )
    parser.add_argument("--gee-days", type=int, default=None, help="Ventana GEE en dias.")
    parser.add_argument(
        "--bluesky-queries",
        default=None,
        help="Queries Bluesky separadas por punto y coma.",
    )
    parser.add_argument("--bluesky-limit", type=int, default=None, help="Posts por query Bluesky.")
    parser.add_argument(
        "--youtube-queries",
        default=None,
        help="Queries YouTube separadas por punto y coma.",
    )
    parser.add_argument("--youtube-max-results", type=int, default=None, help="Videos por query.")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Fusiona .env con argumentos de CLI."""

    config = PipelineConfig.from_env()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.sources:
        config.enabled_sources = sources_from_csv(args.sources)
    if args.disable_cache:
        config.use_cache = False
    if args.gdelt_query:
        config.gdelt_query = args.gdelt_query
    if args.gdelt_timespan:
        config.gdelt_timespan = args.gdelt_timespan
    if args.gdelt_maxrecords is not None:
        config.gdelt_maxrecords = args.gdelt_maxrecords
    if args.google_news_queries:
        config.google_news_queries = split_query_list(
            args.google_news_queries,
            config.google_news_queries,
        )
    if args.opensky_bbox:
        config.opensky_bbox = parse_bbox(args.opensky_bbox, config.opensky_bbox)
    if args.nasa_area:
        config.nasa_firms_area = normalize_nasa_area(args.nasa_area, config.nasa_firms_area)
    if args.nasa_day_range is not None:
        config.nasa_firms_day_range = args.nasa_day_range
    if args.nasa_empty_retry_day_range is not None:
        config.nasa_firms_empty_retry_day_range = args.nasa_empty_retry_day_range
    if args.acled_days is not None:
        config.acled_days = args.acled_days
    if args.acled_limit is not None:
        config.acled_limit = args.acled_limit
    if args.worldpop_year is not None:
        config.worldpop_year = args.worldpop_year
    if args.worldpop_countries:
        config.worldpop_countries = split_query_list(args.worldpop_countries, config.worldpop_countries)
    if args.unhcr_year is not None:
        config.unhcr_year = args.unhcr_year
    if args.unhcr_countries:
        config.unhcr_countries = split_query_list(args.unhcr_countries, config.unhcr_countries)
    if args.hdx_queries:
        config.hdx_queries = split_query_list(args.hdx_queries, config.hdx_queries)
    if args.sentinel_hub_bbox:
        config.sentinel_hub_bbox = normalize_lonlat_bbox(
            args.sentinel_hub_bbox,
            config.sentinel_hub_bbox,
        )
    if args.sentinel_hub_days is not None:
        config.sentinel_hub_days = args.sentinel_hub_days
    if args.osm_bbox:
        config.osm_bbox = normalize_latlon_bbox(args.osm_bbox, config.osm_bbox)
    if args.gee_bbox:
        config.gee_bbox = normalize_lonlat_bbox(args.gee_bbox, config.gee_bbox)
    if args.gee_days is not None:
        config.gee_days = args.gee_days
    if args.bluesky_queries:
        config.bluesky_queries = split_query_list(args.bluesky_queries, config.bluesky_queries)
    if args.bluesky_limit is not None:
        config.bluesky_limit = args.bluesky_limit
    if args.youtube_queries:
        config.youtube_queries = split_query_list(args.youtube_queries, config.youtube_queries)
    if args.youtube_max_results is not None:
        config.youtube_max_results = args.youtube_max_results
    return config


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame | None]:
    """Punto de entrada sin sys.exit(), amigable para notebooks y terminal."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    config = config_from_args(args)
    return run_pipeline(config)


if __name__ == "__main__":
    main()
