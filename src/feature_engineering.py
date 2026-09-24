"""
Feature Engineering Module

Loads processed news data,
creates NLP features,
stores features into feature_store table.
"""

import pandas as pd

from db_connection import get_connection
from logger import logger


def load_processed_data():
    """
    Load unprocessed records from processed_news_articles.

    Returns
    -------
    pandas.DataFrame
    """

    conn = None

    try:

        conn = get_connection()

        query = """
        SELECT *
        FROM processed_news_articles
        WHERE feature_engineered = FALSE;
        """

        df = pd.read_sql(
            query,
            conn
        )

        logger.info(
            f"Loaded {len(df)} processed records"
        )

        print(
            f"Loaded {len(df)} processed records"
        )

        return df

    except Exception as e:

        logger.error(
            f"Failed to load processed data: {e}"
        )

        raise e

    finally:

        if conn:
            conn.close()


def create_features(df):
    """
    Create NLP features.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    try:

        # Number of characters
        df["text_length"] = (
            df["clean_text"]
            .astype(str)
            .apply(len)
        )

        # Number of words
        df["word_count"] = (
            df["clean_text"]
            .astype(str)
            .apply(
                lambda x: len(x.split())
            )
        )

        # Average word length
        df["avg_word_length"] = (
            df["clean_text"]
            .astype(str)
            .apply(
                lambda x:
                sum(
                    len(word)
                    for word in x.split()
                )
                /
                max(
                    len(x.split()),
                    1
                )
            )
        )

        logger.info(
            "Feature engineering completed"
        )

        print(
            "Feature engineering completed"
        )

        return df

    except Exception as e:

        logger.error(
            f"Feature engineering failed: {e}"
        )

        raise e


def save_features(df):
    """
    Save features to PostgreSQL.
    """

    if len(df) == 0:

        logger.info(
            "No records available for feature engineering"
        )

        print(
            "No records available for feature engineering"
        )

        return

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        insert_query = """
        INSERT INTO feature_store
        (
            raw_article_id,
            clean_text,
            text_length,
            word_count,
            avg_word_length
        )
        VALUES
        (
            %s,%s,%s,%s,%s
        );
        """

        records = [

            (
                row["raw_article_id"],
                row["clean_text"],
                int(row["text_length"]),
                int(row["word_count"]),
                float(row["avg_word_length"])
            )

            for _, row in df.iterrows()

        ]

        cursor.executemany(
            insert_query,
            records
        )

        conn.commit()

        logger.info(
            f"{len(records)} feature records inserted"
        )

        print(
            f"{len(records)} feature records inserted"
        )

        # Mark records as feature engineered

        update_query = """
        UPDATE processed_news_articles
        SET feature_engineered = TRUE
        WHERE raw_article_id = %s;
        """

        processed_ids = [

            (
                int(row["raw_article_id"]),
            )

            for _, row in df.iterrows()

        ]

        cursor.executemany(
            update_query,
            processed_ids
        )

        conn.commit()

        logger.info(
            f"{len(processed_ids)} records marked as feature engineered"
        )

        print(
            f"{len(processed_ids)} records marked as feature engineered"
        )

    except Exception as e:

        logger.error(
            f"Failed to save features: {e}"
        )

        raise e

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


if __name__ == "__main__":

    try:

        logger.info(
            "Feature engineering started"
        )

        df = load_processed_data()

        if len(df) == 0:

            print(
                "No processed records found."
            )

        else:

            feature_df = create_features(
                df
            )

            save_features(
                feature_df
            )

            logger.info(
                "Feature engineering completed successfully"
            )

            print(
                "\nFeature engineering completed successfully."
            )

    except Exception as e:

        logger.error(
            f"Pipeline failed: {e}"
        )

        print(
            f"ERROR: {e}"
        )
        