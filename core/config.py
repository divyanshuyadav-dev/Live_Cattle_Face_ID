from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Model weights
YOLO_WEIGHTS = BASE_DIR / "weights" / "yolo26n_cowface.pt"
EMBEDDER_WEIGHTS = BASE_DIR / "weights" / "best.pth"

# Storage paths
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "gallery.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection & Tracking (YOLO26n + ByteTrack)
YOLO_CONF_THRESHOLD = 0.50
YOLO_IOU_THRESHOLD = 0.45
TRACKER_CONFIG = "bytetrack.yaml"

# Quality Gate
MIN_FACE_SIZE = 40          # Minimum width/height in pixels
BLUR_THRESHOLD = 80.0       # Laplacian variance; lower = blurrier

# Embedding (MobileFaceNet)
IMAGE_SIZE = 112
EMBEDDING_DIM = 512

# Open-Set Verification (FAISS + Temporal Voting)
SIMILARITY_THRESHOLD = 0.55  # Initial default; we will calibrate this in Phase 6
CONFIRM_FRAMES = 3           # Consecutive matching frames to lock a track_id
MIN_ENROLL_IMAGES = 3        # Minimum valid face crops required to enroll a cow