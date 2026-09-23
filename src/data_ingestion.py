"""
Data Ingestion Module

Fetches news articles from NewsAPI.org,
stores raw JSON data,
and creates a CSV dataset.
"""

import json
import os
import time
import requests
import pandas as pd
from datetime import datetime
from s3_utils import upload_file
from to_db import insert_data
from dotenv import load_dotenv


# Load environment variables first
load_dotenv()

from config import (
    QUERIES,
    PAGE_SIZE,
    MAX_PAGES,
    NEWS_API_URL,
    NEWS_API_KEY,
    RAW_DATA_PATH,
    RAW_JSON_PATH
)

from logger import logger


def fetch_newsapi_news(query, page):
    """
    Fetch articles from NewsAPI.org
    """

    try:

        params = {
            "q": query,
            "pageSize": PAGE_SIZE,
            "page": page,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt"
        }

        response = requests.get(
            NEWS_API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        articles = data.get(
            "articles",
            []
        )

        logger.info(
            f"{query} | page={page} -> {len(articles)} articles fetched"
        )

        return articles

    except Exception as e:

        logger.error(
            f"Failed to fetch data for query: {query}"
        )

        raise e


def articles_to_dataframe(articles):
    """
    Convert article list to DataFrame
    """

    try:

        records = []

        for article in articles:

            records.append(
                {
                    "source": article.get(
                        "source",
                        {}
                    ).get("name"),

                    "author": article.get(
                        "author"
                    ),

                    "title": article.get(
                        "title"
                    ),

                    "description": article.get(
                        "description"
                    ),

                    "content": article.get(
                        "content"
                    ),

                    "url": article.get(
                        "url"
                    ),

                    "image_url": article.get(
                        "urlToImage"
                    ),

                    "published_at": article.get(
                        "publishedAt"
                    )
                }
            )

        df = pd.DataFrame(records)

        logger.info(
            f"DataFrame created with {len(df)} records"
        )

        return df

    except Exception as e:

        logger.error(
            "Failed to create DataFrame"
        )

        raise e


def save_json(data):

    try:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        json_file = (
            RAW_JSON_PATH.parent
            / f"news_raw_{timestamp}.json"
        )

        with open(
            json_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        logger.info(
            f"JSON saved at {json_file}"
        )

        return json_file

    except Exception as e:

        raise e

def save_csv(df):

    try:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        csv_file = (
            RAW_DATA_PATH.parent
            / f"news_raw_{timestamp}.csv"
        )

        df.to_csv(
            csv_file,
            index=False
        )

        logger.info(
            f"CSV saved at {csv_file}"
        )

        return csv_file

    except Exception as e:

        raise e

if __name__ == "__main__":

    try:

        if not NEWS_API_KEY:

            raise Exception(
                "NEWS_API_KEY not found in .env file"
            )

        RAW_DATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        RAW_JSON_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        all_articles = []

        logger.info(
            "Data ingestion started"
        )

        for query in QUERIES:

            print(
                f"\nFetching -> {query}"
            )

            for page in range(
                1,
                MAX_PAGES + 1
            ):

                articles = fetch_newsapi_news(
                    query=query,
                    page=page
                )

                if len(articles) == 0:
                    break

                all_articles.extend(
                    articles
                )

                print(
                    f"Page {page} -> {len(articles)} articles"
                )

                time.sleep(1)

        print(
            f"\nTotal Raw Articles: {len(all_articles)}"
        )

        json_path = save_json(
            all_articles
        )
        upload_file(
            file_path=str(json_path),
            s3_key=f"raw/{json_path.name}"
        )

        logger.info(
            f"Uploaded JSON to S3: raw/{json_path.name}"
        )
        
        

        df = articles_to_dataframe(
            all_articles
        )

        if len(df) == 0:

            raise Exception(
                "No data collected from API"
            )

        before_count = len(df)

        df.drop_duplicates(
            subset=["url"],
            inplace=True
        )

        df.dropna(
            subset=["title"],
            inplace=True
        )

        df.reset_index(
            drop=True,
            inplace=True
        )

        after_count = len(df)

        print(
            f"Duplicates Removed: {before_count - after_count}"
        )

        print(
            f"Final Dataset Size: {after_count}"
        )

        csv_path = save_csv(df)

        upload_file(
            file_path=str(csv_path),
            s3_key=f"raw/{csv_path.name}"
        )

        logger.info(
            f"Uploaded CSV to S3: raw/{csv_path.name}"
        )
        
        # Insert data into PostgreSQL
        insert_data(df)

        logger.info(
            f"Inserted {len(df)} records into PostgreSQL"
        )
        print(
            f"Inserted {len(df)} records into PostgreSQL"
        )

        print(
            f"\nJSON Saved : {json_path}"
        )

        print(
            f"CSV Saved  : {csv_path}"
        )
      
      
        print(
            "\nData ingestion completed successfully."
        )

    except Exception as e:

        logger.error(
            f"Pipeline failed: {str(e)}"
        )

        print(
            f"ERROR: {e}"
        )