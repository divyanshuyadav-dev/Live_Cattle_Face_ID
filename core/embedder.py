import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.config import EMBEDDER_WEIGHTS, IMAGE_SIZE, EMBEDDING_DIM, DEVICE
from core.mobilefacenet import MobileFaceNet


class CattleEmbedder:
    def __init__(self, checkpoint_path=EMBEDDER_WEIGHTS, device: str = DEVICE):
        self.device = torch.device(device)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False) #change true here later

        if isinstance(ckpt, dict) and "backbone" in ckpt:
            embedding_dim = ckpt.get("embedding_dim", EMBEDDING_DIM)
            state_dict = ckpt["backbone"]
        else:
            embedding_dim = EMBEDDING_DIM
            state_dict = ckpt

        self.model = MobileFaceNet(embedding_size=embedding_dim).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Returns a 1D L2-normalized float32 numpy array of shape (512,)."""
        tensor = self._preprocess(crop_bgr)
        emb = self.model(tensor)
        emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)
