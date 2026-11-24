import cv2
import numpy as np

from core.config import MIN_ENROLL_IMAGES
from core.detector import CowFaceDetector
from core.embedder import CattleEmbedder
from core.gallery import VectorGallery
from core.quality import check_crop_quality
from core.verifier import TrackVerifier


# BGR colors for OpenCV overlays
STATUS_COLORS = {
    "LOCKED": (40, 200, 40),       # Green
    "VERIFYING": (0, 200, 255),    # Yellow
    "UNKNOWN": (40, 40, 230),      # Red
    "LOW_QUALITY": (160, 160, 160) # Gray
}


class CattlePipeline:
    def __init__(self):
        self.detector = CowFaceDetector()
        self.embedder = CattleEmbedder()
        self.gallery = VectorGallery()
        self.verifier = TrackVerifier(self.gallery)

    def enroll_cow(self, label: str, images: list[tuple[str, np.ndarray]]) -> dict:
        """
        Run the exact same detection + quality + embedding pipeline on uploaded images.
        images: list of (filename, image_bgr)
        """
        label = label.strip()
        if not label:
            return {"success": False, "message": "Cow label cannot be empty.", "details": []}

        valid_embeddings = []
        details = []

        for filename, img_bgr in images:
            if img_bgr is None or img_bgr.size == 0:
                details.append({"file": filename, "accepted": False, "reason": "Invalid or unreadable image"})
                continue

            det = self.detector.detect_best(img_bgr)
            if det is None:
                details.append({"file": filename, "accepted": False, "reason": "No cow face detected"})
                continue

            crop = self.detector.crop_face(img_bgr, det["bbox"])
            is_valid, reason, blur_score = check_crop_quality(crop)
            if not is_valid:
                details.append({"file": filename, "accepted": False, "reason": reason})
                continue

            emb = self.embedder.get_embedding(crop)
            valid_embeddings.append(emb)
            details.append({
                "file": filename,
                "accepted": True,
                "reason": f"OK (conf={det['conf']:.2f}, blur={blur_score:.1f})",
            })

        if len(valid_embeddings) < MIN_ENROLL_IMAGES:
            return {
                "success": False,
                "message": f"Need at least {MIN_ENROLL_IMAGES} valid face images, got {len(valid_embeddings)}.",
                "details": details,
            }

        record = self.gallery.enroll(label, valid_embeddings)
        return {
            "success": True,
            "message": f"Enrolled '{record['label']}' using {record['num_samples']} valid images.",
            "record": record,
            "details": details,
        }

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Process a single video frame:
        YOLO26n + ByteTrack -> Fast-path Track Cache -> Quality Gate -> MobileFaceNet -> FAISS -> Draw Overlay.
        """
        annotated = frame_bgr.copy()
        detections = self.detector.track_frame(frame_bgr)

        active_ids = {d["track_id"] for d in detections}
        self.verifier.cleanup_lost_tracks(active_ids)

        results = []

        for det in detections:
            tid = det["track_id"]
            bbox = det["bbox"]

            # 1. Fast path: already locked -> skip crop, quality, embedder, and FAISS
            locked = self.verifier.get_locked(tid)
            if locked is not None:
                res = {"track_id": tid, "bbox": bbox, "status": "LOCKED", **locked}
            else:
                crop = self.detector.crop_face(frame_bgr, bbox)
                is_valid, reason, _ = check_crop_quality(crop)

                if not is_valid:
                    res = {
                        "track_id": tid,
                        "bbox": bbox,
                        "status": "LOW_QUALITY",
                        "entity_id": None,
                        "label": "Low Quality",
                        "score": 0.0,
                    }
                else:
                    emb = self.embedder.get_embedding(crop)
                    ver = self.verifier.verify(tid, emb)
                    res = {"track_id": tid, "bbox": bbox, **ver}

            results.append(res)
            self._draw_box(annotated, res)

        return annotated, results

    def _draw_box(self, frame: np.ndarray, res: dict):
        x1, y1, x2, y2 = res["bbox"]
        status = res["status"]
        color = STATUS_COLORS.get(status, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if status == "LOCKED":
            text = f"#{res['track_id']} {res['label']} ({res['score']:.2f})"
        elif status == "VERIFYING":
            text = f"#{res['track_id']} {res['label']}? ({res['score']:.2f})"
        elif status == "UNKNOWN":
            text = f"#{res['track_id']} Unknown ({res['score']:.2f})"
        else:
            text = f"#{res['track_id']} Low Quality"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(y1 - 8, th + 6)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), color, -1)
        cv2.putText(frame, text, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    def reset_stream(self):
        """Reset tracker and verification cache when switching video sources."""
        self.detector.reset_tracker()
        self.verifier.reset()

    def delete_cow(self, entity_id: int) -> bool:
        deleted = self.gallery.delete_identity(entity_id)
        if deleted:
            self.verifier.reset()
        return deleted