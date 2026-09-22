from db_connection import get_connection
from logger import logging


def fetch_records():

    conn = None
    cur = None

    try:
        logging.info("Fetching records from database")

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM news_articles")

        rows = cur.fetchall()

        for row in rows:
            print(row)

        logging.info(f"Fetched {len(rows)} records successfully")

    except Exception as e:
        logging.error(f"Data fetch failed: {e}")
        print(f" Error: {e}")

    finally:
        if cur:
            cur.close()

        if conn:
            conn.close()

        logging.info("Database connection closed")


if __name__ == "__main__":
    fetch_records()