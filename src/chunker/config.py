"""Configuration schemas for Cascaded Multi-Modal Adaptive Semantic Partitioning (C-MASP)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkerConfig:
    """Master configuration for the IntelligentVideoChunker pipeline."""

    # Duration constraints (in seconds)
    min_chunk_sec: float = 10.0
    max_chunk_sec: float = 45.0
    target_chunk_sec: float = 20.0

    # Tier 0 & Tier 1: Visual kinematics parameters
    sample_fps: float = 2.0
    thumbnail_size: int = 64
    clip_Input_size: int = 224
    zscore_window_sec: float = 10.0
    hard_cut_z_threshold: float = 2.5
    static_flux_threshold: float = 0.015
    min_static_duration_sec: float = 15.0
    suppress_static_segments: bool = True

    # Tier 2: Discourse & acoustic alignment parameters
    speech_boundary_margin_sec: float = 0.25
    speech_snap_window_sec: float = 3.0
    use_audio_rms_fallback: bool = True

    # Tier 3: Semantic regime shift parameters
    semantic_encoder: str = "fast_spatial"  # Options: "fast_spatial" (CPU) or "open_clip" (ViT-B-32)
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    foote_kernel_half_width_sec: float = 4.0
    device: str = "cpu"

    # Tier 4: Dynamic programming objective weights
    weight_intra_variance: float = 3.5
    weight_inter_contrast_bonus: float = 2.4
    weight_path_uniformity: float = 1.2
    weight_length_prior: float = 0.20
    weight_visual_cut_bonus: float = 1.8
    weight_semantic_novelty_bonus: float = 2.0
    weight_speech_interruption_penalty: float = 6.0


@dataclass
class VideoChunk:
    """Represents a single semantically coherent video segment."""

    chunk_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    chunk_type: str  # "ACTIVE_CONTENT" or "STATIC_IDLE"
    boundary_reason: str
    intra_semantic_variance: float
    visual_flux_mean: float
    speech_aligned: bool
    transcript_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.duration_sec, 2),
            "chunk_type": self.chunk_type,
            "boundary_reason": self.boundary_reason,
            "intra_semantic_variance": round(self.intra_semantic_variance, 4),
            "visual_flux_mean": round(self.visual_flux_mean, 4),
            "speech_aligned": self.speech_aligned,
            "transcript_snippet": self.transcript_snippet,
        }


@dataclass
class ChunkingResult:
    """Container for complete video chunking output and telemetry."""

    video_path: str
    video_duration_sec: float
    detected_profile: str
    processing_time_sec: float
    realtime_factor_xrt: float
    chunks: List[VideoChunk] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "video_duration_sec": round(self.video_duration_sec, 2),
            "detected_profile": self.detected_profile,
            "processing_time_sec": round(self.processing_time_sec, 4),
            "realtime_factor_xrt": round(self.realtime_factor_xrt, 2),
            "num_chunks": len(self.chunks),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "chunks": [c.to_dict() for c in self.chunks],
        }