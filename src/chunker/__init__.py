"""Intelligent Video Chunking package (C-MASP architecture)."""

from src.chunker.config import ChunkerConfig, ChunkingResult, VideoChunk
from src.chunker.pipeline import BaselineChunkers, IntelligentVideoChunker

__all__ = [
    "ChunkerConfig",
    "VideoChunk",
    "ChunkingResult",
    "IntelligentVideoChunker",
    "BaselineChunkers",
]
