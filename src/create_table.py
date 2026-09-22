from db_connection import get_connection
from logger import logging


def create_tables():

    conn = None
    cur = None

    try:
        logging.info("Starting table creation process")

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            source VARCHAR(255),
            author TEXT,
            title TEXT,
            description TEXT,
            content TEXT,
            url TEXT UNIQUE,
            image_url TEXT,
            published_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()

        logging.info("news_articles table created successfully")
        print("Table created successfully!")

    except Exception as e:
        logging.error(f"Table creation failed: {e}")
        print(f"Error: {e}")

    finally:
        if cur:
            cur.close()

        if conn:
            conn.close()

        logging.info("Database connection closed")


if __name__ == "__main__":
    create_tables()