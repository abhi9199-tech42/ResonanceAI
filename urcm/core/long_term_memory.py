import json
import os
import pickle
import sqlite3
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from urcm.core.safe_io import safe_load_pickle_bytes


class LongTermMemory:
    """
    Production-Grade Long Term Memory (The Library).

    Architecture:
    - **Metadata Store**: SQLite database (fast filtering, structured queries).
    - **Vector Store**: Memory-mapped NumPy array (fast linear scan) or future-proof Index.
    - **Hybrid Search**: Combine SQL filters (tags, time) with Vector Similarity.

    This is designed to handle thousands of memories efficiently and persist reliably.
    """

    def __init__(self, db_path: str = "urcm_memory.db", vector_dim: int = None):
        self.db_path = db_path
        self.vector_dim = vector_dim

        # Initialize Database
        self._init_db()

        # Cache for vectors (in a real production system, this would be a proper index like HNSW)
        # For < 100k items, exact linear scan with numpy is extremely fast and accurate.
        self._vector_cache: Optional[np.ndarray] = None
        self._id_map: List[int] = [] # Maps index in _vector_cache to DB ID
        self._load_vectors()

    def _init_db(self):
        """Initializes the SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Memory Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                source TEXT,
                timestamp REAL,
                embedding_model TEXT
            )
        ''')

        # Tags Table (Many-to-Many)
        c.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT UNIQUE
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id INTEGER,
                tag_id INTEGER,
                FOREIGN KEY(memory_id) REFERENCES memories(id),
                FOREIGN KEY(tag_id) REFERENCES tags(id)
            )
        ''')

        conn.commit()
        conn.close()

    def _load_vectors(self):
        """Loads all vectors into memory for fast search."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, vector_blob FROM memories")
        rows = c.fetchall()
        conn.close()

        if not rows:
            self._vector_cache = None
            self._id_map = []
            return

        vectors = []
        ids = []
        for r in rows:
            vec = safe_load_pickle_bytes(r[1])
            if vec is None or not hasattr(vec, 'ndim') or vec.ndim != 1:
                continue
            if self.vector_dim is not None:
                if vec.shape[0] < self.vector_dim:
                    pad = self.vector_dim - vec.shape[0]
                    vec = np.pad(vec, (0, pad))
                elif vec.shape[0] > self.vector_dim:
                    vec = vec[:self.vector_dim]
            vectors.append(vec)
            ids.append(r[0])

        if vectors:
            self._vector_cache = np.vstack(vectors)
            self._id_map = ids
            # Ensure vectors are normalized
            norms = np.linalg.norm(self._vector_cache, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 # Avoid div by zero
            self._vector_cache = self._vector_cache / norms

    def add(self, text: str, vector: np.ndarray, tags: List[str] = [], source: str = "user"):
        """Adds a memory with transactional safety."""
        if self.vector_dim is not None and vector is not None and vector.ndim == 1:
            if vector.shape[0] < self.vector_dim:
                pad = self.vector_dim - vector.shape[0]
                vector = np.pad(vector, (0, pad))
            elif vector.shape[0] > self.vector_dim:
                vector = vector[:self.vector_dim]
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        vector_blob = pickle.dumps(vector)
        timestamp = time.time()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            # Insert Memory
            c.execute("INSERT INTO memories (text, vector_blob, source, timestamp) VALUES (?, ?, ?, ?)",
                      (text, vector_blob, source, timestamp))
            memory_id = c.lastrowid

            # Insert Tags
            for tag in tags:
                # Ensure tag exists
                c.execute("INSERT OR IGNORE INTO tags (tag_name) VALUES (?)", (tag,))
                c.execute("SELECT id FROM tags WHERE tag_name = ?", (tag,))
                tag_id = c.fetchone()[0]

                # Link tag
                c.execute("INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)", (memory_id, tag_id))

            conn.commit()
            print(f"[LTM] 💾 Persisted memory #{memory_id}: '{text[:30]}...'")

            # Update Cache (Incremental)
            if self._vector_cache is None:
                self._vector_cache = np.array([vector])
                self._id_map = [memory_id]
            else:
                # Ensure runtime cache vectors match target dimension
                v = vector
                if self.vector_dim is not None and v.shape[0] != self.vector_dim:
                    if v.shape[0] < self.vector_dim:
                        v = np.pad(v, (0, self.vector_dim - v.shape[0]))
                    else:
                        v = v[:self.vector_dim]
                self._vector_cache = np.vstack([self._vector_cache, v])
                self._id_map.append(memory_id)

        except Exception as e:
            conn.rollback()
            print(f"[LTM] ❌ Error saving memory: {e}")
        finally:
            conn.close()

    def retrieve(self, query_vector: np.ndarray, k: int = 5, min_similarity: float = 0.0,
                 filter_tags: List[str] = [], filter_source: str = None) -> List[Tuple[Dict, float]]:
        """
        Retrieves memories using Hybrid Search (Vector + Metadata Filters).
        """
        if self._vector_cache is None:
            return []

        # 1. Vector Search (Fast Global Scan)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        similarities = np.dot(self._vector_cache, query_vector)

        # Get top 3*k candidates first (to allow for filtering fallout)
        # If we have filters, we need to fetch more candidates or filter first.
        # For simplicity/speed in this prototype, we'll scan all, then filter.
        # Optimisation: If N is huge, filter IDs in SQL first, then search only those vectors.

        candidate_indices = np.argsort(similarities)[::-1] # Descending

        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        count = 0
        for idx in candidate_indices:
            sim_score = float(similarities[idx])
            if sim_score < min_similarity:
                break

            mem_id = self._id_map[idx]

            # Check Metadata Filters (Lazy Load)
            # Build SQL Query dynamically for this specific ID
            query = "SELECT m.id, m.text, m.source, m.timestamp FROM memories m WHERE m.id = ?"
            params = [mem_id]

            if filter_source:
                query += " AND m.source = ?"
                params.append(filter_source)

            c.execute(query, params)
            row = c.fetchone()

            if not row:
                continue # Filtered out by source

            # Check Tags if needed
            if filter_tags:
                c.execute('''
                    SELECT t.tag_name FROM tags t
                    JOIN memory_tags mt ON t.id = mt.tag_id
                    WHERE mt.memory_id = ?
                ''', (mem_id,))
                mem_tags = {r[0] for r in c.fetchall()}
                if not set(filter_tags).issubset(mem_tags):
                    continue # Missing required tags
            else:
                # Fetch tags for result display
                c.execute('''
                    SELECT t.tag_name FROM tags t
                    JOIN memory_tags mt ON t.id = mt.tag_id
                    WHERE mt.memory_id = ?
                ''', (mem_id,))
                mem_tags = [r[0] for r in c.fetchall()]

            # Valid Result
            result_meta = {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "timestamp": row[3],
                "tags": list(mem_tags)
            }
            results.append((result_meta, sim_score))
            count += 1
            if count >= k:
                break

        conn.close()
        return results

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM memories")
        count = c.fetchone()[0]
        conn.close()
        return {"total_memories": count, "vector_dim": self._vector_cache.shape[1] if self._vector_cache is not None else 0}
