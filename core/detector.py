import numpy as np
from ultralytics import YOLO
from core.config import (
    YOLO_WEIGHTS,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    TRACKER_CONFIG,
    DEVICE,
)


class CowFaceDetector:
    def __init__(self, weights_path=YOLO_WEIGHTS):
        self.model = YOLO(str(weights_path))
        self.device = DEVICE

    def detect_best(self, image_bgr: np.ndarray) -> dict | None:
        """
        Stateless detection for Enrollment UI.
        Returns the highest-confidence bounding box or None.
        """
        results = self.model.predict(
            source=image_bgr,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf = float(boxes.conf[best_idx].item())

        return {"bbox": (int(x1), int(y1), int(x2), int(y2)), "conf": conf}

    def track_frame(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Stateful detection + ByteTrack for Recognition UI.
        Returns a list of tracked cow faces with persistent track_ids.
        """
        results = self.model.track(
            source=frame_bgr,
            # imgsz=320,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            tracker=TRACKER_CONFIG,
            persist=True,
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        detections = []
        xyxys = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, tid in zip(xyxys, confs, track_ids):
            detections.append({
                "track_id": int(tid),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(conf),
            })

        return detections

    def reset_tracker(self):
        """Clear ByteTrack state when starting a new video or camera stream."""
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers = []

    @staticmethod
    def crop_face(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Safely clamp bounding box to frame dimensions and return crop."""
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image_bgr[y1:y2, x1:x2]