"""
Data Ingestion Module

Supports:
1. Manual ingestion: python src/data_ingestion.py
2. AWS Lambda ingestion: lambda_handler(event, context)

Fetches English news articles from NewsAPI, uploads raw JSON/CSV to S3,
and inserts unique article URLs into PostgreSQL.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
import psycopg
import requests
from dotenv import load_dotenv

# Load .env for local execution. Lambda environment variables take precedence.
load_dotenv()

from config import (
    QUERIES,
    PAGE_SIZE,
    MAX_PAGES,
    NEWS_API_URL,
    NEWS_API_KEY,
    RAW_DATA_PATH,
    RAW_JSON_PATH,
    AWS_DEFAULT_REGION,
    S3_BUCKET_NAME,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)

from s3_utils import upload_file
from to_db import insert_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REGION = os.getenv("AWS_REGION") or AWS_DEFAULT_REGION or "ap-south-1"
LAMBDA_SECRET_ARN = os.getenv("INGESTION_SECRET_ARN")
LAMBDA_BUCKET = os.getenv("S3_BUCKET_NAME") or S3_BUCKET_NAME
NEWS_API_TIMEOUT = 25


def is_lambda_environment():
    """Return True when running inside AWS Lambda."""
    return bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


def load_lambda_secrets():
    """Read NewsAPI and PostgreSQL credentials from Secrets Manager."""
    if not LAMBDA_SECRET_ARN:
        raise RuntimeError(
            "INGESTION_SECRET_ARN environment variable is required in Lambda."
        )

    client = boto3.client("secretsmanager", region_name=REGION)
    response = client.get_secret_value(SecretId=LAMBDA_SECRET_ARN)

    secret_string = response.get("SecretString")
    if not secret_string:
        raise RuntimeError("The ingestion secret has no SecretString.")

    secret = json.loads(secret_string)

    required = [
        "NEWS_API_KEY",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]
    missing = [key for key in required if not secret.get(key)]
    if missing:
        raise RuntimeError(
            "Missing required fields in the ingestion secret: "
            + ", ".join(missing)
        )

    return secret


def fetch_newsapi_news(query, page, api_key):
    """Fetch one page of articles from NewsAPI."""
    params = {
        "q": query,
        "pageSize": PAGE_SIZE,
        "page": page,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "publishedAt",
    }

    response = requests.get(
        NEWS_API_URL,
        params=params,
        timeout=NEWS_API_TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI returned an unsuccessful status for query {query!r}."
        )

    articles = payload.get("articles", [])
    logger.info(
        "Fetched articles: query=%r page=%s count=%s",
        query,
        page,
        len(articles),
    )
    return articles


def collect_articles(api_key):
    """Collect articles across the configured queries and pages."""
    all_articles = []

    for query in QUERIES:
        for page in range(1, MAX_PAGES + 1):
            articles = fetch_newsapi_news(query, page, api_key)

            if not articles:
                break

            all_articles.extend(articles)
            time.sleep(1)

    return all_articles


def articles_to_dataframe(articles):
    """Convert NewsAPI article dictionaries into the database schema."""
    records = []

    for article in articles:
        source = article.get("source") or {}

        records.append(
            {
                "source": source.get("name"),
                "author": article.get("author"),
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "url": article.get("url"),
                "image_url": article.get("urlToImage"),
                "published_at": article.get("publishedAt"),
            }
        )

    columns = [
        "source",
        "author",
        "title",
        "description",
        "content",
        "url",
        "image_url",
        "published_at",
    ]
    df = pd.DataFrame(records, columns=columns)

    if not df.empty:
        df = df.dropna(subset=["title", "url"])
        df = df[df["title"].astype(str).str.strip().ne("")]
        df = df[df["url"].astype(str).str.strip().ne("")]
        df = df.drop_duplicates(subset=["url"], keep="first")
        df = df.reset_index(drop=True)

    logger.info("Prepared %s unique articles.", len(df))
    return df


def save_json(data, output_dir):
    """Write the raw article response to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = output_dir / f"news_raw_{timestamp}_{uuid.uuid4().hex[:8]}.json"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return file_path


def save_csv(df, output_dir):
    """Write the processed article dataset to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = output_dir / f"news_raw_{timestamp}_{uuid.uuid4().hex[:8]}.csv"

    df.to_csv(file_path, index=False)
    return file_path


def upload_to_s3(file_path, s3_key):
    """Upload through the Lambda role in AWS, or the existing helper locally."""
    if not LAMBDA_BUCKET:
        raise RuntimeError("S3_BUCKET_NAME is not configured.")

    if is_lambda_environment():
        client = boto3.client("s3", region_name=REGION)
        client.upload_file(
            str(file_path),
            LAMBDA_BUCKET,
            s3_key,
        )
    else:
        # Preserve existing local behavior.
        upload_file(
            file_path=str(file_path),
            s3_key=s3_key,
        )

    logger.info("Uploaded s3://%s/%s", LAMBDA_BUCKET, s3_key)


def insert_data_lambda(df, secret):
    """Insert articles in Lambda and propagate database failures."""
    if df.empty:
        return 0

    conn = None
    try:
        conn = psycopg.connect(
            host=secret["DB_HOST"],
            port=int(secret.get("DB_PORT", "5432")),
            dbname=secret["DB_NAME"],
            user=secret["DB_USER"],
            password=secret["DB_PASSWORD"],
            connect_timeout=10,
            sslmode="require",
        )

        records = [
            (
                row.source,
                row.author,
                row.title,
                row.description,
                row.content,
                row.url,
                row.image_url,
                row.published_at,
            )
            for row in df.itertuples(index=False)
        ]

        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO news_articles
                (
                    source, author, title, description,
                    content, url, image_url, published_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING
                """,
                records,
            )
            inserted_count = max(cursor.rowcount, 0)

        conn.commit()
        logger.info("Inserted %s new database records.", inserted_count)
        return inserted_count

    except Exception:
        if conn is not None:
            conn.rollback()
        logger.exception("PostgreSQL insertion failed.")
        raise

    finally:
        if conn is not None:
            conn.close()


def run_ingestion(lambda_mode=False):
    """Run the complete ingestion workflow."""
    if lambda_mode:
        secret = load_lambda_secrets()
        api_key = secret["NEWS_API_KEY"]
        output_dir = Path("/tmp")
    else:
        secret = None
        api_key = NEWS_API_KEY
        output_dir = RAW_DATA_PATH.parent

    if not api_key:
        raise RuntimeError("NEWS_API_KEY is not configured.")

    logger.info("News ingestion started. lambda_mode=%s", lambda_mode)

    all_articles = collect_articles(api_key)
    if not all_articles:
        raise RuntimeError("NewsAPI returned no articles.")

    df = articles_to_dataframe(all_articles)
    if df.empty:
        raise RuntimeError("No usable articles remain after cleaning.")

    json_path = save_json(all_articles, output_dir)
    csv_path = save_csv(df, output_dir)

    # Upload both artifacts before reporting a successful ingestion.
    run_suffix = uuid.uuid4().hex[:8]
    upload_to_s3(json_path, f"raw/{json_path.name}")
    upload_to_s3(csv_path, f"raw/{csv_path.name}")

    if lambda_mode:
        inserted_count = insert_data_lambda(df, secret)
    else:
        # Existing manual path is preserved.
        insert_data(df)
        # The current local helper suppresses DB errors; its behavior is unchanged.
        inserted_count = None

    result = {
        "status": "success",
        "articles_fetched": len(all_articles),
        "unique_articles": len(df),
        "database_rows_inserted": inserted_count,
        "json_s3_key": f"raw/{json_path.name}",
        "csv_s3_key": f"raw/{csv_path.name}",
    }

    logger.info("News ingestion completed: %s", json.dumps(result))
    return result


def lambda_handler(event, context):
    """AWS Lambda entry point invoked by EventBridge."""
    try:
        result = run_ingestion(lambda_mode=True)
        return result
    except Exception:
        logger.exception("Scheduled news ingestion failed.")
        raise


if __name__ == "__main__":
    try:
        result = run_ingestion(lambda_mode=False)
        print(json.dumps(result, indent=2))
    except Exception:
        logger.exception("Manual news ingestion failed.")
        raise
