"""
Project Configuration File
"""

import os
from pathlib import Path

# Base directory

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR

# Data paths

RAW_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "news_raw.csv"
)

RAW_JSON_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "news_raw.json"
)

# NewsAPI.org Configuration

NEWS_API_URL = (
    "https://newsapi.org/v2/everything"
)

NEWS_API_KEY = os.getenv(
    "NEWS_API_KEY"
)

# News Queries

QUERIES = [
    "artificial intelligence",
    "machine learning",
    "technology",
    "business",
    "finance",
    "stock market",
    "economy",
    "health",
    "sports",
    "politics",
    "cybersecurity",
    "startup",
    "climate change",
    "world news",
    "global news"
]

# API Settings

PAGE_SIZE = 100
MAX_PAGES = 1


# AWS DB Configuration
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")