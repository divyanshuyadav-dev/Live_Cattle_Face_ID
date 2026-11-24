from collections import Counter, defaultdict
import numpy as np

from core.config import CONFIRM_FRAMES, SIMILARITY_THRESHOLD
from core.gallery import VectorGallery


class TrackVerifier:
    def __init__(
        self,
        gallery: VectorGallery,
        threshold: float = SIMILARITY_THRESHOLD,
        confirm_frames: int = CONFIRM_FRAMES,
    ):
        self.gallery = gallery
        self.threshold = threshold
        self.confirm_frames = confirm_frames

        # track_id -> {"entity_id": int, "label": str, "score": float}
        self.locked_tracks: dict[int, dict] = {}
        self.vote_buffer: dict[int, list[tuple[int, str, float]]] = defaultdict(list)

    def get_locked(self, track_id: int) -> dict | None:
        """Fast-path check before running MobileFaceNet."""
        return self.locked_tracks.get(track_id)

    def verify(self, track_id: int, embedding: np.ndarray) -> dict:
        """
        Query FAISS and apply temporal consensus voting.
        Returns dict with keys: status, entity_id, label, score
        """
        if track_id in self.locked_tracks:
            cached = self.locked_tracks[track_id]
            return {"status": "LOCKED", **cached}

        entity_id, label, score = self.gallery.search(embedding, self.threshold)
        if entity_id is None:
            return {"status": "UNKNOWN", "entity_id": None, "label": "Unknown", "score": score}

        self.vote_buffer[track_id].append((entity_id, label, score))

        if len(self.vote_buffer[track_id]) >= self.confirm_frames:
            votes = self.vote_buffer[track_id]
            top_id, count = Counter(v[0] for v in votes).most_common(1)[0]

            if count >= self.confirm_frames:
                matching_scores = [v[2] for v in votes if v[0] == top_id]
                avg_score = float(np.mean(matching_scores))
                top_label = next(v[1] for v in votes if v[0] == top_id)

                result = {"entity_id": top_id, "label": top_label, "score": avg_score}
                self.locked_tracks[track_id] = result
                del self.vote_buffer[track_id]
                return {"status": "LOCKED", **result}

            # Keep sliding window bounded if votes are mixed
            self.vote_buffer[track_id] = votes[-self.confirm_frames:]

        return {"status": "VERIFYING", "entity_id": entity_id, "label": label, "score": score}

    def cleanup_lost_tracks(self, active_track_ids: set[int]):
        """Prune tracks that ByteTrack is no longer reporting."""
        for tid in list(self.locked_tracks.keys()):
            if tid not in active_track_ids:
                del self.locked_tracks[tid]
        for tid in list(self.vote_buffer.keys()):
            if tid not in active_track_ids:
                del self.vote_buffer[tid]

    def reset(self):
        """Clear all track states when switching video sources or deleting an identity."""
        self.locked_tracks.clear()
        self.vote_buffer.clear()