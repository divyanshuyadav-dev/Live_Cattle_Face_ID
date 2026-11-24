import sqlite3
import threading
from datetime import datetime, timezone

import faiss
import numpy as np

from core.config import DB_PATH, EMBEDDING_DIM, SIMILARITY_THRESHOLD


class VectorGallery:
    def __init__(self, db_path=DB_PATH, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self.id_to_label: dict[int, str] = {}
        self.rebuild_index()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    num_samples INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(1, self.dim)
        faiss.normalize_L2(arr)
        return arr

    def enroll(self, label: str, embeddings: list[np.ndarray]) -> dict:
        """Compute L2-normalized centroid from multiple face embeddings, save to SQLite, and index in FAISS."""
        if not embeddings:
            raise ValueError("Cannot enroll with an empty embedding list.")

        centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm_vec = self._normalize(centroid)
        created_at = datetime.now(timezone.utc).isoformat()

        with self.lock:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO identities (label, num_samples, created_at, embedding) VALUES (?, ?, ?, ?)",
                    (label.strip(), len(embeddings), created_at, norm_vec.tobytes()),
                )
                entity_id = int(cur.lastrowid)

            self.index.add_with_ids(norm_vec, np.array([entity_id], dtype=np.int64))
            self.id_to_label[entity_id] = label.strip()

        return {
            "id": entity_id,
            "label": label.strip(),
            "num_samples": len(embeddings),
            "created_at": created_at,
        }

    def rebuild_index(self):
        """Load all embeddings from SQLite into FAISS."""
        with self.lock:
            self.index.reset()
            self.id_to_label.clear()

            rows = self.conn.execute("SELECT id, label, embedding FROM identities").fetchall()
            if not rows:
                return

            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            vecs = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            self.index.add_with_ids(vecs, ids)

            for r in rows:
                self.id_to_label[int(r["id"])] = r["label"]

    def search(self, query_emb: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> tuple[int | None, str | None, float]:
        """Return (entity_id, label, score) if score >= threshold, else (None, None, score)."""
        with self.lock:
            if self.index.ntotal == 0:
                return None, None, 0.0

            q = self._normalize(query_emb)
            scores, ids = self.index.search(q, k=1)

            best_score = float(scores[0][0])
            best_id = int(ids[0][0])

            if best_id != -1 and best_score >= threshold:
                return best_id, self.id_to_label.get(best_id, f"ID_{best_id}"), best_score
            return None, None, best_score

    def list_identities(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, label, num_samples, created_at FROM identities ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_identity(self, entity_id: int) -> bool:
        with self.lock:
            with self.conn:
                cur = self.conn.execute("DELETE FROM identities WHERE id = ?", (entity_id,))
                deleted = cur.rowcount > 0

        if deleted:
            self.rebuild_index()
        return deleted