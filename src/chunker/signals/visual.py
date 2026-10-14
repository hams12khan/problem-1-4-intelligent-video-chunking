"""Tier 1: Fast Visual Kinematics, Edge Flux, and CFAR Median/MAD Cut Peak Detector."""

from typing import Optional, Tuple
import cv2
import numpy as np


class VisualSignalExtractor:
    """Extracts HSV color flux, Sobel structural edge flux, and CFAR local Z-score cut peaks."""

    def __init__(
        self,
        sample_fps: float = 2.0,
        thumbnail_size: int = 64,
        zscore_window_sec: float = 10.0,
    ):
        self.sample_fps = sample_fps
        self.thumbnail_size = thumbnail_size
        self.zscore_window_sec = zscore_window_sec

    def decode_sparse_frames(
        self, video_path: str, decode_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Decodes video at sparse target FPS using hardware-friendly OpenCV grab/retrieve.

        Args:
            video_path: Path to input video file.
            decode_size: Spatial resolution for decoded frames (defaults to self.thumbnail_size,
                         or 224 when ViT encoder is enabled).

        Returns:
            timestamps: (T,) array of frame timestamps in seconds.
            frames_bgr: (T, H, W, 3) uint8 array of downsampled frames.
            duration_sec: Total video duration in seconds.
        """
        target_sz = decode_size if decode_size is not None else self.thumbnail_size
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or np.isnan(native_fps):
            native_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / native_fps if total_frames > 0 else 0.0

        stride = max(1, int(round(native_fps / self.sample_fps)))
        timestamps = []
        frames = []

        frame_idx = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % stride == 0:
                ret_dec, frame = cap.retrieve()
                if ret_dec and frame is not None:
                    small = cv2.resize(
                        frame,
                        (target_sz, target_sz),
                        interpolation=cv2.INTER_AREA,
                    )
                    frames.append(small)
                    timestamps.append(frame_idx / native_fps)
            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Zero frames decoded from {video_path}")

        actual_duration = max(duration_sec, timestamps[-1] + (1.0 / self.sample_fps))
        return (
            np.array(timestamps, dtype=np.float32),
            np.stack(frames, axis=0),
            float(actual_duration),
        )

    def compute_visual_signals(
        self, frames_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes composite visual flux and CFAR Median/MAD local Z-score novelty.

        Uses robust Median Absolute Deviation (MAD) in the sliding window so rapid
        montage cuts do not inflate background standard deviation and mask each other.
        """
        num_frames = len(frames_bgr)
        if num_frames <= 1:
            return (
                np.zeros(num_frames, dtype=np.float32),
                np.zeros(num_frames, dtype=np.float32),
            )

        hist_features = []
        edge_maps = []
        thumb_sz = self.thumbnail_size

        for frame in frames_bgr:
            if frame.shape[0] != thumb_sz or frame.shape[1] != thumb_sz:
                proc_frame = cv2.resize(
                    frame, (thumb_sz, thumb_sz), interpolation=cv2.INTER_AREA
                )
            else:
                proc_frame = frame

            hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
            )
            hist = hist.flatten().astype(np.float32)
            norm = np.linalg.norm(hist) + 1e-8
            hist_features.append(hist / norm)

            gray = (
                cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            )
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            edge_maps.append(mag)

        hist_matrix = np.stack(hist_features, axis=0)  # (T, 128)
        edge_tensor = np.stack(edge_maps, axis=0)      # (T, H, W)

        # Cosine distance between consecutive HSV histograms
        dot_prod = np.sum(hist_matrix[1:] * hist_matrix[:-1], axis=1)
        hist_diff = np.clip(1.0 - dot_prod, 0.0, 1.0)

        # Structural edge change ratio
        edge_l1 = np.mean(np.abs(edge_tensor[1:] - edge_tensor[:-1]), axis=(1, 2))

        # Combined visual flux
        visual_flux = np.zeros(num_frames, dtype=np.float32)
        visual_flux[1:] = 0.65 * hist_diff + 0.35 * np.clip(edge_l1 * 2.0, 0.0, 1.0)

        # Robust CFAR sliding-window Z-score via Median and MAD
        win_radius = max(3, int(round(self.zscore_window_sec * self.sample_fps / 2.0)))
        raw_zscore = np.zeros(num_frames, dtype=np.float32)

        for t in range(1, num_frames):
            lo = max(0, t - win_radius)
            hi = min(num_frames, t + win_radius + 1)
            local_slice = visual_flux[lo:hi]
            med = float(np.median(local_slice))
            mad = float(np.median(np.abs(local_slice - med)))
            robust_sigma = max(1.4826 * mad, float(np.std(local_slice)) * 0.5, 0.015)
            z = (visual_flux[t] - med) / robust_sigma
            raw_zscore[t] = max(0.0, z)

        # Local 3-tap Non-Maximum Suppression (NMS) so a single shot cut is a sharp impulse
        nms_zscore = np.zeros(num_frames, dtype=np.float32)
        for t in range(1, num_frames - 1):
            if raw_zscore[t] >= raw_zscore[t - 1] and raw_zscore[t] >= raw_zscore[t + 1]:
                nms_zscore[t] = raw_zscore[t]
        if num_frames > 1 and raw_zscore[-1] > raw_zscore[-2]:
            nms_zscore[-1] = raw_zscore[-1]

        norm_zscore = np.clip(nms_zscore / 3.5, 0.0, 1.0)
        return visual_flux, norm_zscore