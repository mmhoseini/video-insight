import pandas as pd
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_values
from typing import List, Dict, Any
import threading
from queue import Queue
from tqdm import tqdm
from threading import Event
import boto3
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm
from io import BytesIO
from psycopg2.extras import execute_values
import queue
import threading
from dataclasses import dataclass
from threading import Event
import psycopg2
from psycopg2.pool import ThreadedConnectionPool


class TitleEmbeddingsLoader:
    def __init__(
        self,
        df: pd.DataFrame,
        db_config,
        db_workers: int = 4,
        batch_size: int = 1000,
        queue_size: int = 2000
    ):
        self.df = df
        self.db_workers = db_workers
        self.batch_size = batch_size
        
        # Initialize queue
        self.db_queue = Queue(maxsize=queue_size)
        
        # Initialize database connection pool
        self.db_pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=db_workers + 1,
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password
        )
        
        # Control flags
        self.stop_event = Event()
        self.error_event = Event()
        
        # Progress tracking
        self.total_items = len(df)
        self.processed_items = 0
        self.progress_lock = threading.Lock()
        self.pbar = None
    
    def _db_worker(self):
        """Worker function to insert embeddings into database"""
        conn = self.db_pool.getconn()
        batch = []
        
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.db_queue.get(timeout=1)
                    if item is None:
                        break
                    
                    batch.append((item['video_id'], item['embedding'].tolist(), None))
                    
                    if len(batch) >= self.batch_size:
                        self._process_batch(conn, batch)
                        batch = []
                    
                    self.db_queue.task_done()
                    
                except Queue.Empty:
                    if batch:  # Process remaining items
                        self._process_batch(conn, batch)
                        batch = []
                    continue
                    
        except Exception as e:
            print(f"DB worker error: {str(e)}")
            self.error_event.set()
        finally:
            if batch:
                try:
                    self._process_batch(conn, batch)
                except Exception as e:
                    print(f"Error in final batch processing: {str(e)}")
            self.db_pool.putconn(conn)

    def _process_batch(self, conn, batch):
        """Process a batch of items"""
        if not batch:
            return
            
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO video_embeddings 
                (video_id, title_embedding, thumbnail_embedding)
                VALUES %s
                ON CONFLICT (video_id) DO UPDATE
                SET title_embedding = EXCLUDED.title_embedding
                """,
                batch
            )
            conn.commit()
        
        with self.progress_lock:
            self.processed_items += len(batch)
            if self.pbar is not None:
                self.pbar.update(len(batch))
    
    def populate_embeddings(self):
        """Main function to populate title embeddings"""
        try:
            print(f"Processing {self.total_items} videos")
            self.pbar = tqdm(total=self.total_items, desc="Processing")
            
            # Start DB workers
            db_threads = []
            for _ in range(self.db_workers):
                t = threading.Thread(target=self._db_worker)
                t.daemon = True
                t.start()
                db_threads.append(t)
            
            # Feed DB queue with all data first
            for _, row in self.df.iterrows():
                if self.error_event.is_set():
                    break
                self.db_queue.put(row, timeout=30)
            
            # Signal workers to stop after all data is queued
            for _ in range(self.db_workers):
                self.db_queue.put(None)
            
            # Wait for queue to empty
            self.db_queue.join()
            
            # Set stop event and cleanup
            self.stop_event.set()
            for t in db_threads:
                t.join(timeout=1)
            
        except Exception as e:
            print(f"Error in populate_embeddings: {str(e)}")
            self.error_event.set()
            
        finally:
            self.stop_event.set()
            if hasattr(self, 'db_pool'):
                self.db_pool.closeall()
            if self.pbar is not None:
                self.pbar.close()
            
            if self.error_event.is_set():
                print("\nProcessing stopped due to errors")
            else:
                print(f"\nProcessed {self.processed_items} videos successfully")



@dataclass
class EmbeddingItem:
    video_id: str
    s3_path: str
    embedding: Optional[np.ndarray] = None


class ThumbnailEmbeddingsLoader:
    def __init__(
        self,
        df: pd.DataFrame,
        db_config,
        s3_client=None,
        download_workers: int = 8,
        db_workers: int = 2,
        batch_size: int = 1000,
        queue_size: int = 2000
    ):
        self.df = df
        self.s3_client = s3_client or boto3.client('s3')
        self.download_workers = download_workers
        self.db_workers = db_workers
        self.batch_size = batch_size
        
        # Initialize queues with reasonable sizes
        self.download_queue = queue.Queue(maxsize=queue_size)
        self.db_queue = queue.Queue(maxsize=queue_size)
        
        # Initialize database connection pool
        self.db_pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=db_workers + 1,  # +1 for main thread
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password
        )
        
        # Control flags
        self.stop_event = Event()
        self.error_event = Event()
        
        # Progress tracking
        self.total_items = len(df)
        self.processed_items = 0
        self.progress_lock = threading.Lock()
        self.pbar = None
    
    def _download_worker(self):
        """Worker function to download embeddings from S3 and send directly to DB queue"""
        while not self.stop_event.is_set():
            try:
                item = self.download_queue.get(timeout=1)
                if item is None:
                    break
                
                try:
                    bucket = item.s3_path.split('/')[2]
                    key = '/'.join(item.s3_path.split('/')[3:])
                    response = self.s3_client.get_object(Bucket=bucket, Key=key)
                    
                    array_buffer = BytesIO(response['Body'].read())
                    embedding_array = np.load(array_buffer)
                    item.embedding = embedding_array[0]  # Take first vector
                    
                    # Send directly to DB queue
                    self.db_queue.put(item, timeout=30)
                    
                except Exception as e:
                    print(f"Error downloading {item.s3_path}: {str(e)}")
                finally:
                    self.download_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Download worker error: {str(e)}")
                self.error_event.set()
                break
    
    def _db_worker(self):
        """Worker function to insert embeddings into database"""
        conn = self.db_pool.getconn()
        batch = []
        last_commit_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Force batch processing if it's been too long
                    if batch and (time.time() - last_commit_time > 10):
                        self._process_batch(conn, batch)
                        batch = []
                        last_commit_time = time.time()
                    
                    # Get item from queue
                    item = self.db_queue.get(timeout=1)
                    if item is None:
                        break
                    
                    batch.append((item.video_id, None, item.embedding.tolist()))
                    
                    # Process batch if full
                    if len(batch) >= self.batch_size:
                        self._process_batch(conn, batch)
                        batch = []
                        last_commit_time = time.time()
                    
                    self.db_queue.task_done()
                    
                except queue.Empty:
                    if batch:  # Process remaining items
                        self._process_batch(conn, batch)
                        batch = []
                    continue
                    
        except Exception as e:
            print(f"DB worker error: {str(e)}")
            self.error_event.set()
        finally:
            # Process any remaining items
            if batch:
                try:
                    self._process_batch(conn, batch)
                except Exception as e:
                    print(f"Error in final batch processing: {str(e)}")
            self.db_pool.putconn(conn)

    def _process_batch(self, conn, batch):
        """Process a batch of items"""
        if not batch:
            return
            
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO video_embeddings 
                (video_id, title_embedding, thumbnail_embedding)
                VALUES %s
                ON CONFLICT (video_id) DO UPDATE
                SET thumbnail_embedding = EXCLUDED.thumbnail_embedding
                """,
                batch
            )
            conn.commit()
        
        # Update progress
        with self.progress_lock:
            self.processed_items += len(batch)
            if self.pbar is not None:
                self.pbar.update(len(batch))
    
    def _get_missing_videos(self) -> set:
        """Get video IDs that don't have embeddings in the database"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT v.video_id
                    FROM videos v
                    LEFT JOIN video_embeddings ve ON v.video_id = ve.video_id
                    WHERE ve.video_id IS NULL
                """)
                missing_video_ids = {row[0] for row in cur.fetchall()}
            return missing_video_ids
        finally:
            self.db_pool.putconn(conn)
    
    def populate_embeddings(self):
        """Main function to populate embeddings using pipeline architecture"""
        try:
            # Get missing videos
            missing_video_ids = self._get_missing_videos()
            videos_to_process = self.df[self.df['video_id'].isin(missing_video_ids)]
            
            if videos_to_process.empty:
                print("No new videos to process")
                return
            
            total_videos = len(videos_to_process)
            print(f"Processing {total_videos} videos")
            
            # Initialize progress bar
            self.pbar = tqdm(total=total_videos, desc="Processing")
            
            # Start all workers
            all_threads = []
            
            # Start download workers
            for _ in range(self.download_workers):
                t = threading.Thread(target=self._download_worker)
                t.daemon = True
                t.start()
                all_threads.append(t)
            
            # Start DB workers
            for _ in range(self.db_workers):
                t = threading.Thread(target=self._db_worker)
                t.daemon = True
                t.start()
                all_threads.append(t)
            
            # Feed download queue with all data first
            for _, row in videos_to_process.iterrows():
                if self.error_event.is_set():
                    break
                
                item = EmbeddingItem(
                    video_id=row['video_id'],
                    s3_path=row['embedding_s3_path']
                )
                self.download_queue.put(item, timeout=30)
            
            # Signal download workers to stop after all data is queued
            for _ in range(self.download_workers):
                self.download_queue.put(None)
            
            # Wait for download queue to empty
            self.download_queue.join()
            
            # Signal DB workers to stop
            for _ in range(self.db_workers):
                self.db_queue.put(None)
            
            # Wait for DB queue to empty
            self.db_queue.join()
            
            # Set stop event and cleanup
            self.stop_event.set()
            for t in all_threads:
                t.join(timeout=1)
            
        except Exception as e:
            print(f"Error in populate_embeddings: {str(e)}")
            self.error_event.set()
            
        finally:
            self.stop_event.set()
            if hasattr(self, 'db_pool'):
                self.db_pool.closeall()
            if self.pbar is not None:
                self.pbar.close()
            
            if self.error_event.is_set():
                print("\nProcessing stopped due to errors")
            else:
                print(f"\nProcessed {self.processed_items} videos successfully")