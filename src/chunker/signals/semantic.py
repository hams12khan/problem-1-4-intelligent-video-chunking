"""Tier 3: Sparse Semantic Embeddings (Fast Spatial & GPU-Batched OpenCLIP) + Foote's Novelty."""

import cv2
import numpy as np


class SemanticSignalExtractor:
    """Computes balanced frame embeddings and detects regime shifts via Foote's Kernel."""

    def __init__(
        self,
        encoder_type: str = "fast_spatial",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        foote_half_width_sec: float = 4.0,
        sample_fps: float = 2.0,
        device: str = "cpu",
    ):
        self.encoder_type = encoder_type
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.foote_half_width_sec = foote_half_width_sec
        self.sample_fps = sample_fps
        self.device = device
        self._clip_model = None

    def _load_open_clip_if_needed(self):
        if self._clip_model is not None:
            return
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained, device=self.device
        )
        model.eval()
        self._clip_model = model

    def compute_embeddings(self, frames_bgr: np.ndarray) -> np.ndarray:
        """Extracts L2-normalized semantic embeddings (T, D) for sampled frames.

        Supports two modes:
        - 'fast_spatial': Training-free Block-Balanced Multi-Scale Spatial-Lab-DCT Descriptor (~450 FPS on CPU).
        - 'open_clip': Frozen pre-trained ViT-B-32 vision encoder with direct GPU tensor batching.
        """
        if self.encoder_type == "open_clip":
            try:
                return self._compute_clip_embeddings(frames_bgr)
            except Exception:
                return self._compute_fast_spatial_embeddings(frames_bgr)
        return self._compute_fast_spatial_embeddings(frames_bgr)

    @staticmethod
    def _normalize_block(vec: np.ndarray, weight: float = 1.0) -> np.ndarray:
        """Normalizes an individual feature sub-block to unit L2 norm with safe fallback."""
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            return (vec / norm) * weight
        return np.zeros_like(vec, dtype=np.float32)

    def _compute_fast_spatial_embeddings(self, frames_bgr: np.ndarray) -> np.ndarray:
        """Constructs a 192-D Block-Balanced Multi-Scale Spatial-Chroma + AC-DCT descriptor."""
        embeddings = []
        for frame in frames_bgr:
            if frame.shape[0] != 64 or frame.shape[1] != 64:
                frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
            lab_centered = lab.copy()
            lab_centered[:, :, 0] -= 0.5
            lab_centered[:, :, 1:] -= 128.0 / 255.0

            # Block 1: 4x4 Spatial Pyramid chromaticity & luminance layout (48 dims)
            grid_4x4 = cv2.resize(
                lab_centered, (4, 4), interpolation=cv2.INTER_AREA
            ).flatten()
            b1 = self._normalize_block(grid_4x4, weight=0.60)

            # Block 2: 4x4 Regional spatial variance / detail (48 dims)
            h = (frame.shape[0] // 4) * 4
            w = (frame.shape[1] // 4) * 4
            if h > 0 and w > 0:
                blocks = lab_centered[:h, :w].reshape(4, h // 4, 4, w // 4, 3)
                grid_std = np.std(blocks, axis=(1, 3)).flatten()
            else:
                grid_std = np.zeros(48, dtype=np.float32)
            b2 = self._normalize_block(grid_std, weight=0.45)

            # Block 3: 8x8 2D-DCT structural AC frequency coefficients (64 dims)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray_32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            dct_block = cv2.dct(gray_32 - float(np.mean(gray_32)))[:8, :8].copy()
            dct_block[0, 0] = 0.0
            b3 = self._normalize_block(dct_block.flatten(), weight=0.50)

            # Block 4: Global HSV color distribution (32 dims)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            hist_hsv = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
            b4 = self._normalize_block(hist_hsv, weight=0.55)

            vec = np.concatenate([b1, b2, b3, b4]).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec = vec / norm
            else:
                vec = np.zeros(192, dtype=np.float32)
                vec[0] = 1.0
            embeddings.append(vec)

        return np.stack(embeddings, axis=0)

    def _compute_clip_embeddings(self, frames_bgr: np.ndarray) -> np.ndarray:
        """Vectorized GPU/CPU batch inference for OpenCLIP ViT-B-32 (eliminates PIL loop overhead)."""
        import torch
        import torch.nn.functional as F

        self._load_open_clip_if_needed()
        dev = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        use_fp16 = dev.type == "cuda"

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev, dtype=torch.float32).view(1, 3, 1, 1)

        all_embs = []
        batch_size = 128 if use_fp16 else 32
        num_frames = len(frames_bgr)

        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                chunk_bgr = frames_bgr[i : i + batch_size]
                # Convert BGR uint8 (B, H, W, 3) -> RGB float32 tensor (B, 3, H, W) in [0, 1]
                chunk_rgb = np.ascontiguousarray(chunk_bgr[:, :, :, ::-1])
                tensor = torch.from_numpy(chunk_rgb).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
                if tensor.shape[2] != 224 or tensor.shape[3] != 224:
                    tensor = F.interpolate(tensor, size=(224, 224), mode="bicubic", align_corners=False)
                tensor = (tensor - mean) / std

                if use_fp16:
                    with torch.amp.autocast("cuda"):
                        feats = self._clip_model.encode_image(tensor)
                else:
                    feats = self._clip_model.encode_image(tensor)

                feats = feats.float()
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
                all_embs.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embs, axis=0)

    def compute_foote_novelty(self, embeddings: np.ndarray) -> np.ndarray:
        """Computes Foote's Gaussian-Tapered Checkerboard Kernel Novelty along the diagonal."""
        num_frames = len(embeddings)
        half_w = max(2, int(round(self.foote_half_width_sec * self.sample_fps)))
        if num_frames < 2 * half_w + 1:
            novelty = np.zeros(num_frames, dtype=np.float32)
            if num_frames > 1:
                dots = np.sum(embeddings[1:] * embeddings[:-1], axis=1)
                novelty[1:] = np.clip(1.0 - dots, 0.0, 1.0)
            return novelty

        k_size = 2 * half_w
        coords = np.linspace(-1.0, 1.0, k_size)
        x, y = np.meshgrid(coords, coords)
        gaussian = np.exp(-(x**2 + y**2) / (2.0 * (0.5**2)))
        signs = np.ones((k_size, k_size), dtype=np.float32)
        signs[:half_w, half_w:] = -1.0
        signs[half_w:, :half_w] = -1.0
        kernel = signs * gaussian
        kernel /= np.sum(np.abs(kernel)) + 1e-8

        novelty = np.zeros(num_frames, dtype=np.float32)
        for t in range(half_w, num_frames - half_w):
            window_emb = embeddings[t - half_w : t + half_w]
            local_ssm = window_emb @ window_emb.T
            score = float(np.sum(local_ssm * kernel))
            novelty[t] = max(0.0, score)

        p98 = (
            float(np.percentile(novelty, 98))
            if num_frames > 5
            else float(np.max(novelty) if num_frames > 0 else 1.0)
        )
        norm_scale = max(p98, 0.15)
        novelty = np.clip(novelty / norm_scale, 0.0, 1.0)
        return novelty
