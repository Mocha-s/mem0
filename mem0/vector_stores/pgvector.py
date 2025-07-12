import json
import logging
from typing import List, Optional
import threading
from contextlib import contextmanager

from pydantic import BaseModel

try:
    import psycopg2
    from psycopg2.extras import execute_values
    from psycopg2 import pool
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    raise ImportError("The 'psycopg2' library is required. Please install it using 'pip install psycopg2'.")

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # 只显示错误级别的日志


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    metadata: Optional[dict]
    
    @property
    def payload(self):
        """向后兼容：提供 payload 属性来访问 metadata"""
        return self.metadata


class PGVector(VectorStoreBase):
    def __init__(
        self,
        dbname,
        collection_name,
        embedding_model_dims,
        user,
        password,
        host,
        port,
        diskann,
        hnsw,
        max_connections=20,
        min_connections=1,
    ):
        """
        Initialize the optimized PGVector database with connection pooling.

        Args:
            dbname (str): Database name
            collection_name (str): Collection name
            embedding_model_dims (int): Dimension of the embedding vector
            user (str): Database user
            password (str): Database password
            host (str, optional): Database host
            port (int, optional): Database port
            diskann (bool, optional): Use DiskANN for faster search
            hnsw (bool, optional): Use HNSW for faster search
            max_connections (int): Maximum connections in pool
            min_connections (int): Minimum connections in pool
        """
        self.collection_name = collection_name
        self.use_diskann = diskann
        self.use_hnsw = hnsw
        self.embedding_model_dims = embedding_model_dims
        
        # 创建连接池
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        
        self._lock = threading.Lock()
        
        # 初始化集合
        collections = self.list_cols()
        if collection_name not in collections:
            self.create_col(embedding_model_dims)

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit=True):
        """获取数据库游标的上下文管理器"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
                raise
            finally:
                cursor.close()

    def create_col(self, embedding_model_dims):
        """
        Create a new collection (table in PostgreSQL) with improved error handling.
        """
        try:
            with self.get_cursor() as cur:
                # 创建扩展
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # 创建表
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.collection_name} (
                        id UUID PRIMARY KEY,
                        vector vector({embedding_model_dims}),
                        payload JSONB
                    );
                """
                )

                # 创建索引
                if self.use_diskann and embedding_model_dims < 2000:
                    # 检查vectorscale扩展
                    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vectorscale'")
                    if cur.fetchone():
                        cur.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS {self.collection_name}_diskann_idx
                            ON {self.collection_name}
                            USING diskann (vector);
                        """
                        )
                elif self.use_hnsw:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx
                        ON {self.collection_name}
                        USING hnsw (vector vector_cosine_ops)
                    """
                    )
                
                logger.info(f"Collection {self.collection_name} created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create collection {self.collection_name}: {e}")
            raise

    def insert(self, vectors, payloads=None, ids=None):
        """
        Insert vectors into a collection with improved error handling.
        """
        try:
            logger.debug(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
            json_payloads = [json.dumps(payload) for payload in payloads]

            data = [(id, vector, payload) for id, vector, payload in zip(ids, vectors, json_payloads)]
            
            with self.get_cursor() as cur:
                execute_values(
                    cur,
                    f"""INSERT INTO {self.collection_name} (id, vector, payload) VALUES %s 
                        ON CONFLICT (id) DO UPDATE SET 
                        vector = EXCLUDED.vector, 
                        payload = EXCLUDED.payload""",
                    data,
                )
                logger.debug(f"Successfully upserted {len(vectors)} vectors")
                
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    def search(self, query, vectors, limit=5, filters=None):
        """
        Search for similar vectors with improved error handling.
        """
        try:
            filter_conditions = []
            filter_params = []

            if filters:
                for k, v in filters.items():
                    filter_conditions.append("payload->>%s = %s")
                    filter_params.extend([k, str(v)])

            filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

            with self.get_cursor(commit=False) as cur:
                cur.execute(
                    f"""
                    SELECT id, vector <=> %s::vector AS distance, payload
                    FROM {self.collection_name}
                    {filter_clause}
                    ORDER BY distance
                    LIMIT %s
                """,
                    (vectors, *filter_params, limit),
                )

                results = cur.fetchall()
                return [OutputData(id=str(r[0]), score=float(r[1]), metadata=r[2]) for r in results]
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def delete(self, vector_id):
        """
        Delete a vector by ID with improved error handling.
        """
        try:
            with self.get_cursor() as cur:
                cur.execute(f"DELETE FROM {self.collection_name} WHERE id = %s", (vector_id,))
                logger.debug(f"Deleted vector {vector_id}")
                
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            raise

    def update(self, vector_id, vector=None, payload=None):
        """
        Update a vector and its payload with improved error handling.
        """
        try:
            with self.get_cursor() as cur:
                if vector:
                    cur.execute(
                        f"UPDATE {self.collection_name} SET vector = %s WHERE id = %s",
                        (vector, vector_id),
                    )
                if payload:
                    cur.execute(
                        f"UPDATE {self.collection_name} SET payload = %s WHERE id = %s",
                        (json.dumps(payload), vector_id),
                    )
                logger.debug(f"Updated vector {vector_id}")
                
        except Exception as e:
            logger.error(f"Failed to update vector {vector_id}: {e}")
            raise

    def get(self, vector_id) -> OutputData:
        """
        Retrieve a vector by ID with improved error handling.
        """
        try:
            with self.get_cursor(commit=False) as cur:
                cur.execute(
                    f"SELECT id, vector, payload FROM {self.collection_name} WHERE id = %s",
                    (vector_id,),
                )
                result = cur.fetchone()
                if not result:
                    return None
                return OutputData(id=str(result[0]), score=None, metadata=result[2])
                
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            raise

    def list_cols(self) -> List[str]:
        """
        List all collections with improved error handling.
        """
        try:
            with self.get_cursor(commit=False) as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                return [row[0] for row in cur.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    def delete_col(self):
        """Delete a collection with improved error handling."""
        try:
            with self.get_cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
                logger.info(f"Collection {self.collection_name} deleted")
                
        except Exception as e:
            logger.error(f"Failed to delete collection {self.collection_name}: {e}")
            raise

    def col_info(self):
        """
        Get information about a collection with improved error handling.
        """
        try:
            with self.get_cursor(commit=False) as cur:
                cur.execute(
                    f"""
                    SELECT 
                        table_name, 
                        (SELECT COUNT(*) FROM {self.collection_name}) as row_count,
                        (SELECT pg_size_pretty(pg_total_relation_size('{self.collection_name}'))) as total_size
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                """,
                    (self.collection_name,),
                )
                result = cur.fetchone()
                return {"name": result[0], "count": result[1], "size": result[2]}
                
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def list(self, filters=None, limit=100):
        """
        List all vectors in a collection with improved error handling.
        """
        try:
            filter_conditions = []
            filter_params = []

            if filters:
                for k, v in filters.items():
                    filter_conditions.append("payload->>%s = %s")
                    filter_params.extend([k, str(v)])

            filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

            query = f"""
                SELECT id, vector, metadata
                FROM {self.collection_name}
                {filter_clause}
                LIMIT %s
            """

            with self.get_cursor(commit=False) as cur:
                cur.execute(query, (*filter_params, limit))
                results = cur.fetchall()
                return [[OutputData(id=str(r[0]), score=None, metadata=r[2]) for r in results]]
                
        except Exception as e:
            logger.error(f"Failed to list vectors: {e}")
            raise

    def health_check(self):
        """
        健康检查方法 - 使用独立连接避免影响其他操作
        """
        try:
            # 使用独立的连接进行健康检查
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    if result and result[0] == 1:
                        # 显式提交，确保连接状态清洁
                        conn.commit()
                        return {"status": "healthy", "message": "Vector store is operational"}
                    else:
                        conn.rollback()
                        return {"status": "unhealthy", "message": "Unexpected health check result"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "message": str(e)}

    def reset(self):
        """Reset the index by deleting and recreating it with improved error handling."""
        try:
            logger.warning(f"Resetting index {self.collection_name}...")
            self.delete_col()
            self.create_col(self.embedding_model_dims)
            logger.info(f"Index {self.collection_name} reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset index: {e}")
            raise

    def reset_connection_pool(self):
        """重置连接池以清除所有连接状态"""
        try:
            logger.info("重置向量存储连接池...")
            
            # 关闭现有连接池
            if hasattr(self, 'connection_pool') and self.connection_pool:
                self.connection_pool.closeall()
            
            # 重新创建连接池
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                1,  # min_connections
                20,  # max_connections  
                dbname=self.connection_pool.dsn.split()[0].split('=')[1],
                user=self.connection_pool.dsn.split()[1].split('=')[1],
                password=self.connection_pool.dsn.split()[2].split('=')[1],
                host=self.connection_pool.dsn.split()[3].split('=')[1],
                port=self.connection_pool.dsn.split()[4].split('=')[1]
            )
            
            logger.info("连接池重置完成")
            return {"status": "success", "message": "Connection pool reset successfully"}
            
        except Exception as e:
            logger.error(f"连接池重置失败: {e}")
            return {"status": "error", "message": str(e)}

    def close(self):
        """关闭连接池"""
        if hasattr(self, 'connection_pool') and self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Connection pool closed")

    def __del__(self):
        """
        Close the connection pool when the object is deleted.
        """
        self.close()