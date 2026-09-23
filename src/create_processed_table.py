from db_connection import get_connection

try:

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    ALTER TABLE news_articles
    ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT FALSE;
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS processed_news_articles
    (
        id SERIAL PRIMARY KEY,
        raw_article_id INTEGER,
        clean_text TEXT,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()

    print("Tables updated successfully!")

except Exception as e:

    print(f"Error: {e}")

finally:

    if cursor:
        cursor.close()

    if conn:
        conn.close()