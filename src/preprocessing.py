"""
Data Preprocessing Module

Loads raw news articles from PostgreSQL,
cleans text data,
stores processed records into
processed_news_articles table,
and marks raw records as processed.
"""
import re
import pandas as pd

from db_connection import get_connection
from logger import logger


def load_data_from_db():
    """
    Load unprocessed records from PostgreSQL.

    Returns
    -------
    pandas.DataFrame
    """

    conn = None

    try:

        conn = get_connection()

        query = """
        SELECT *
        FROM news_articles
        WHERE processed = FALSE;
        """

        df = pd.read_sql(
            query,
            conn
        )

        logger.info(
            f"Loaded {len(df)} records from DB"
        )

        print(
            f"Loaded {len(df)} records from DB"
        )

        return df

    except Exception as e:

        logger.error(
            f"Failed to load data from DB: {e}"
        )

        raise e

    finally:

        if conn:
            conn.close()

            logger.info(
                "DB connection closed"
            )


def clean_text(text):
    """
    Clean text data.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    try:

        if pd.isna(text):
            return ""

        text = str(text)

        # Remove URLs
        text = re.sub(
            r"http\S+|www\S+",
            "",
            text
        )

        # Remove special characters
        text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text
        )

        # Convert to lowercase
        text = text.lower()

        # Remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    except Exception as e:

        logger.error(
            f"Text cleaning failed: {e}"
        )

        return ""


def preprocess_dataframe(df):
    """
    Preprocess dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    try:

        df["text"] = (
            df["title"].fillna("")
            + " "
            + df["description"].fillna("")
            + " "
            + df["content"].fillna("")
        )

        df["clean_text"] = df["text"].apply(
            clean_text
        )

        before_count = len(df)

        df.drop_duplicates(
            subset=["url"],
            inplace=True
        )

        df = df[
            df["clean_text"].str.strip() != ""
        ]

        df.reset_index(
            drop=True,
            inplace=True
        )

        after_count = len(df)

        logger.info(
            f"Duplicates removed: {before_count - after_count}"
        )

        print(
            f"Processed records: {after_count}"
        )

        return df

    except Exception as e:

        logger.error(
            f"Preprocessing failed: {e}"
        )

        raise e


def insert_processed_data(df):
    """
    Insert processed records into
    processed_news_articles table.
    """

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        insert_query = """
        INSERT INTO processed_news_articles
        (
            raw_article_id,
            clean_text
        )
        VALUES
        (
            %s,
            %s
        );
        """

        records = [

            (
                row["id"],
                row["clean_text"]
            )

            for _, row in df.iterrows()

        ]

        cursor.executemany(
            insert_query,
            records
        )

        conn.commit()

        logger.info(
            f"{len(records)} processed records inserted"
        )

        print(
            f"{len(records)} processed records inserted"
        )

    except Exception as e:

        logger.error(
            f"Failed to insert processed data: {e}"
        )

        raise e

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


def mark_as_processed(ids):
    """
    Mark raw records as processed.
    """

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        update_query = """
        UPDATE news_articles
        SET processed = TRUE
        WHERE id = ANY(%s);
        """

        cursor.execute(
            update_query,
            (ids,)
        )

        conn.commit()

        logger.info(
            f"{len(ids)} records marked as processed"
        )

        print(
            f"{len(ids)} records marked as processed"
        )

    except Exception as e:

        logger.error(
            f"Failed to update records: {e}"
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
            "Preprocessing started"
        )

        df = load_data_from_db()

        if len(df) == 0:

            print(
                "No unprocessed records found."
            )

        else:

            processed_df = preprocess_dataframe(
                df
            )

            insert_processed_data(
                processed_df
            )

            mark_as_processed(
                processed_df["id"].tolist()
            )

            logger.info(
                "Preprocessing completed successfully"
            )

            print(
                "\nPreprocessing completed successfully."
            )

    except Exception as e:

        logger.error(
            f"Pipeline failed: {e}"
        )

        print(
            f"ERROR: {e}"
        )