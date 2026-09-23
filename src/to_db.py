from db_connection import get_connection
import pandas as pd

def insert_data(df):
    """
     Insert news records into PostgreSQL.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing news articles

    Returns
    -------
    None
    """
    
    conn=None
    cursor=None
    
    
    try:
        #Create DB connection
        conn = get_connection()
        
        # Create a cursor object
        cursor = conn.cursor()
        
        #SQL Insert query
        insert_query = """
        INSERT INTO news_articles
        (
            source,
            author,
            title,
            description,
            content,
            url,
            image_url,
            published_at
        )
        VALUES
        (
            %s,%s,%s,%s,%s,%s,%s,%s
        )
        ON CONFLICT (url)
        DO NOTHING;
        """
        
        #Convert DataFrame to list of tuples
        records = [
            (
                row["source"],
                row["author"],
                row["title"],
                row["description"],
                row["content"],
                row["url"],
                row["image_url"],
                row["published_at"]
            )
            for _, row in df.iterrows()
        ]
        
        #Bulk insert records
        cursor.executemany(
            insert_query,
            records
        )
        
        
        #Commit changes
        conn.commit()
        
        print(
            f"Inserted {cursor.rowcount} records to DB Successfully."
        )
        
    except Exception as e:
        
        print(
            f"Failed to insert data into DB: {e}"
        )
        
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        
        #close connection
        if conn:
            conn.close()
            
        print(
            "DB Connection closed."
        )
        
        
