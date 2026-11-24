import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.config import BASE_DIR, SIMILARITY_THRESHOLD, UPLOAD_DIR
from core.pipeline import CattlePipeline

app = FastAPI(title="Cattle Face Verification System")

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize models, FAISS gallery, and ByteTrack state once at startup
pipeline = CattlePipeline()


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/cows")
def list_cows():
    return {
        "cows": pipeline.gallery.list_identities(),
        "threshold": pipeline.verifier.threshold,
    }


@app.post("/api/enroll")
async def enroll_cow(
    label: str = Form(...),
    files: list[UploadFile] = File(...),
):
    images: list[tuple[str, np.ndarray]] = []

    for f in files:
        raw_bytes = await f.read()
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        images.append((f.filename or "unnamed.jpg", img_bgr))

    result = pipeline.enroll_cow(label, images)
    return result


@app.delete("/api/cows/{entity_id}")
def delete_cow(entity_id: int):
    deleted = pipeline.delete_cow(entity_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Identity not found.")
    return {"success": True, "deleted_id": entity_id}


@app.post("/api/threshold")
def update_threshold(value: float = Query(..., ge=0.1, le=0.99)):
    pipeline.verifier.threshold = value
    pipeline.verifier.reset()
    return {"threshold": pipeline.verifier.threshold}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOAD_DIR / saved_name

    with open(saved_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    return {"filename": saved_name}


def generate_mjpeg(source):
    pipeline.reset_stream()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _ = pipeline.process_frame(frame)

            # Overlay live FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} | Thresh: {pipeline.verifier.threshold:.2f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/api/stream")
def stream_video(
    source: str = Query("webcam", pattern="^(webcam|video)$"),
    filename: str | None = Query(None),
    camera_index: int = Query(0, ge=0),
):
    if source == "webcam":
        cap_source = camera_index
    else:
        if not filename:
            raise HTTPException(status_code=400, detail="Missing video filename.")
        video_path = (UPLOAD_DIR / Path(filename).name).resolve()
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Uploaded video not found.")
        cap_source = str(video_path)

    return StreamingResponse(
        generate_mjpeg(cap_source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )