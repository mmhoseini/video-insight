import psycopg2
from psycopg2.extras import execute_values
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class DatabaseConfig:
    host: str = os.environ.get("DB_HOST")
    port: int = int(os.environ.get("DB_PORT"))
    database: str = os.environ.get("DB_NAME")
    user: str = os.environ.get("DB_USER")
    password: str = os.environ.get("DB_PASSWORD")

class YouTubeVectorDB:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None
        self.setup_connection()
    
    def setup_connection(self):
        """Establish database connection and create necessary tables"""
        self.conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
        
        with self.conn.cursor() as cur:
            # Enable vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create main videos table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id VARCHAR(20) PRIMARY KEY,
                    ucid VARCHAR(40) NOT NULL,
                    author VARCHAR(100) NOT NULL,
                    uploaded TIMESTAMP NOT NULL,
                    date_index DATE NOT NULL,
                    duration INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    v7_score INTEGER,
                    v30_score FLOAT,
                    v7_average FLOAT,
                    performance_index FLOAT,
                    metric FLOAT,
                    subscribers INTEGER,
                    l1_category VARCHAR(50),
                    l2_cat VARCHAR(100),
                    predictions INTEGER,
                    prediction_class VARCHAR(50),
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    s3_path TEXT,
                    pk TEXT[],
                    pk_num INTEGER,
                    pk_perf_ratio FLOAT[],
                    title_length_words INTEGER,
                    title_length_chars INTEGER,
                    emotions TEXT[],
                    emotion_scores FLOAT[],
                    aesthetic_score FLOAT,
                    dte_length FLOAT,
                    dte_punctuation FLOAT,
                    dte_emojis FLOAT,
                    dte_capitalization FLOAT,
                    dte_phrases FLOAT,
                    dte_total FLOAT,
                    category_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create embeddings table with category_index
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_embeddings (
                    video_id VARCHAR(20) PRIMARY KEY REFERENCES videos(video_id),
                    category_index INTEGER,
                    title_embedding vector(768),
                    thumbnail_embedding vector(1152) 
                )
                """)
            
            # Create indices
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON videos(category_index);
                CREATE INDEX IF NOT EXISTS idx_embeddings_category ON video_embeddings(category_index);
            """)
            
            self.conn.commit()

    def insert_video(self, video_data: Dict[str, Any]):
        """Insert a video record into the videos table"""
        with self.conn.cursor() as cur:
            columns = video_data.keys()
            values = [video_data[column] for column in columns]
            
            query = f"""
                INSERT INTO videos ({', '.join(columns)})
                VALUES ({', '.join(['%s'] * len(columns))})
                ON CONFLICT (video_id) DO UPDATE
                SET ({', '.join(columns)}) = ({', '.join(['EXCLUDED.' + col for col in columns])})
            """
            
            cur.execute(query, values)
            self.conn.commit()

    def insert_embeddings(self, video_id: str, title_embedding: List[float], 
                         thumbnail_embedding: List[float], category_index: int = None):
        """Insert embeddings for a video"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO video_embeddings (video_id, title_embedding, thumbnail_embedding, category_index)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (video_id) DO UPDATE
                SET title_embedding = EXCLUDED.title_embedding,
                    thumbnail_embedding = EXCLUDED.thumbnail_embedding,
                    category_index = EXCLUDED.category_index
            """, (video_id, title_embedding, thumbnail_embedding, category_index))
            self.conn.commit()

    def find_similar_videos(self, title_embedding: Optional[List[float]] = None,
                       thumbnail_embedding: Optional[List[float]] = None,
                       threshold: float = 0.8,
                       limit: int = 10) -> List[str]:
        """Find similar videos using either title or thumbnail embeddings.
        If title_embedding is provided, uses that for similarity search.
        If only thumbnail_embedding is provided, uses that instead.
        At least one embedding must be provided.
        Returns list of video_ids ordered by similarity."""
        
        if title_embedding is None and thumbnail_embedding is None:
            raise ValueError("At least one of title_embedding or thumbnail_embedding must be provided")
        
        with self.conn.cursor() as cur:
            if title_embedding is not None:
                cur.execute("""
                    SELECT video_id 
                    FROM video_embeddings
                    WHERE 1 - (title_embedding <=> %s::vector) >= %s
                    ORDER BY title_embedding <=> %s::vector
                    LIMIT %s
                """, (title_embedding, threshold, title_embedding, limit))
            else:
                cur.execute("""
                    SELECT video_id
                    FROM video_embeddings 
                    WHERE 1 - (thumbnail_embedding <=> %s::vector) >= %s
                    ORDER BY thumbnail_embedding <=> %s::vector
                    LIMIT %s
                """, (thumbnail_embedding, threshold, thumbnail_embedding, limit))
                
            return [row[0] for row in cur.fetchall()]
    
    def get_features(self, video_ids: List[str]) -> pd.DataFrame:
        """Fetch all features from videos table for given video_ids.
        Returns a pandas DataFrame with all columns from videos table."""
        
        if not video_ids:
            return pd.DataFrame()
            
        with self.conn.cursor() as cur:
            # Get column names
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'videos'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in cur.fetchall()]
            
            # Fetch video data
            query = """
                SELECT * FROM videos 
                WHERE video_id = ANY(%s)
                ORDER BY ARRAY_POSITION(%s, video_id)
            """
            cur.execute(query, (video_ids, video_ids))
            data = cur.fetchall()
            
            return pd.DataFrame(data, columns=columns)

    def batch_insert_videos(self, videos_data: List[Dict[str, Any]]):
        """Batch insert multiple video records"""
        if not videos_data:
            return
            
        columns = videos_data[0].keys()
        values = [[video[column] for column in columns] for video in videos_data]
        
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO videos ({', '.join(columns)})
                VALUES %s
                ON CONFLICT (video_id) DO UPDATE
                SET ({', '.join(columns)}) = ({', '.join(['EXCLUDED.' + col for col in columns])})
                """,
                values
            )
            self.conn.commit()

    def close(self):
        """Close the database connection"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()