"""
Create Feature Store Table
"""

from db_connection import get_connection
from logger import logger


def create_feature_store_table():

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        create_query = """
        CREATE TABLE IF NOT EXISTS feature_store
        (
            id SERIAL PRIMARY KEY,
            raw_article_id INTEGER UNIQUE NOT NULL,
            clean_text TEXT,
            text_length INTEGER,
            word_count INTEGER,
            unique_word_count INTEGER,
            avg_word_length FLOAT,
            sentiment_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor.execute(
            create_query
        )

        conn.commit()

        logger.info(
            "feature_store table created successfully."
        )

    except Exception as e:

        logger.error(
            f"Error creating table: {e}"
        )

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


if __name__ == "__main__":

    create_feature_store_table()