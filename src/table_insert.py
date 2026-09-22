from db_connection import get_connection
from logger import logging


def insert_test_record():

    conn = None
    cur = None

    try:
        logging.info("Starting test data insertion")

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO news_articles (
            source,
            title,
            url
        )
        VALUES (%s, %s, %s)
        """,
        (
            "Test Source",
            "Test Title",
            "https://test.com"
        ))

        conn.commit()

        logging.info("Test record inserted successfully")
        print("Test record inserted successfully!")

    except Exception as e:
        logging.error(f"Insertion failed: {e}")
        print(f"Error: {e}")

    finally:
        if cur:
            cur.close()

        if conn:
            conn.close()

        logging.info("Database connection closed")


if __name__ == "__main__":
    insert_test_record()