"""
Project Configuration File
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

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

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

#AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


# Processed Data Path

PROCESSED_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "news_processed.csv"
)