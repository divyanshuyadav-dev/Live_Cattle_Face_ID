import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from core.detector import CowFaceDetector
from core.embedder import CattleEmbedder


def collect_embeddings(val_dir: Path, use_yolo: bool = False):
    embedder = CattleEmbedder()
    detector = CowFaceDetector() if use_yolo else None

    embeddings = []
    labels = []

    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    for class_idx, cow_dir in enumerate(class_dirs):
        for img_path in cow_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if use_yolo:
                det = detector.detect_best(img)
                if det is None:
                    continue
                img = detector.crop_face(img, det["bbox"])

            emb = embedder.get_embedding(img)
            embeddings.append(emb)
            labels.append(class_idx)

    return np.vstack(embeddings), np.array(labels)


def calibrate(val_dir: str, use_yolo: bool = False):
    embs, labels = collect_embeddings(Path(val_dir), use_yolo=use_yolo)
    n = len(labels)
    if n < 2:
        raise ValueError("Not enough valid images found in validation directory.")

    # Pairwise cosine similarity matrix (embeddings are already L2-normalized)
    sim_matrix = embs @ embs.T
    rows, cols = np.triu_indices(n, k=1)

    pair_scores = sim_matrix[rows, cols]
    pair_labels = (labels[rows] == labels[cols]).astype(int)

    auc = roc_auc_score(pair_labels, pair_scores)
    fpr, tpr, thresholds = roc_curve(pair_labels, pair_scores)
    fnr = 1.0 - tpr

    # 1. Equal Error Rate (EER): where FAR == FRR
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    tau_eer = float(thresholds[eer_idx])

    # 2. High-Security Operating Point: TAR @ FAR <= 1%
    far_mask = fpr <= 0.01
    if np.any(far_mask):
        idx_1pct = int(np.where(far_mask)[0][-1])
        tar_at_1pct = float(tpr[idx_1pct])
        tau_1pct = float(thresholds[idx_1pct])
    else:
        tar_at_1pct, tau_1pct = 0.0, tau_eer

    genuine = pair_scores[pair_labels == 1]
    impostor = pair_scores[pair_labels == 0]

    print("=" * 55)
    print(f"Images evaluated   : {n} ({len(np.unique(labels))} unseen cows)")
    print(f"Genuine pairs      : {len(genuine)} | Mean: {genuine.mean():.4f} | Std: {genuine.std():.4f}")
    print(f"Impostor pairs     : {len(impostor)} | Mean: {impostor.mean():.4f} | Std: {impostor.std():.4f}")
    print("-" * 55)
    print(f"ROC-AUC            : {auc:.4f}")
    print(f"EER                : {eer * 100:.2f}%  --> Recommended tau (EER):      {tau_eer:.4f}")
    print(f"TAR @ FAR <= 1%    : {tar_at_1pct * 100:.2f}% --> Strict tau (FAR <= 1%):    {tau_1pct:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True, help="Path to unseen validation folder (e.g., cattle_dataset_128/val)")
    parser.add_argument("--use-yolo", action="store_true", help="Run YOLO26n crop before embedding (use if val images are uncropped)")
    args = parser.parse_args()

    calibrate(args.val_dir, use_yolo=args.use_yolo)