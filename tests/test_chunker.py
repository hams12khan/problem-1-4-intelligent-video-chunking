"""Deterministic unit test suite for C-MASP Intelligent Video Chunker (V3)."""

import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker.config import ChunkerConfig, VideoChunk
from src.chunker.evaluation.metrics import ChunkingEvaluator
from src.chunker.pipeline import BaselineChunkers, IntelligentVideoChunker
from src.chunker.signals.discourse import DiscourseSignalExtractor, UtteranceInterval
from src.chunker.signals.semantic import SemanticSignalExtractor


def _get_video_writer(video_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Helper to open VideoWriter with multi-codec fallback for cross-platform stability."""
    for codec in ["mp4v", "MJPG", "XVID", "avc1"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not initialize cv2.VideoWriter for {video_path}")


def create_synthetic_test_video_and_srt(video_path: str, srt_path: str, fps: int = 10):
    """Generates a 60-second synthetic video with 3 distinct visual scenes and an SRT file.

    - Scene 1 (0s - 20s): Solid blue presentation slide with static rectangle.
    - Scene 2 (20s - 42s): Bright red/green checkerboard pattern.
    - Scene 3 (42s - 60s): Moving white circle on dark background.
    """
    width, height = 128, 128
    writer = _get_video_writer(video_path, float(fps), width, height)

    total_frames = 60 * fps
    for f in range(total_frames):
        sec = f / float(fps)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if sec < 20.0:
            img[:, :] = (200, 40, 20)
            cv2.rectangle(img, (20, 20), (100, 80), (255, 255, 255), -1)
        elif sec < 42.0:
            img[:, :] = (20, 180, 40)
            img[::16, ::16] = (10, 10, 240)
            cv2.circle(img, (64, 64), 30, (0, 220, 220), -1)
        else:
            img[:, :] = (30, 30, 30)
            cx = int(20 + ((sec - 42.0) / 18.0) * 88)
            cv2.circle(img, (cx, 64), 22, (250, 250, 250), -1)
        writer.write(img)

    writer.release()

    srt_content = """1
00:00:02,000 --> 00:00:18,500
Welcome to the system architecture presentation covering our first module.

2
00:00:21,000 --> 00:00:27,500
Now we transition to the second scene demonstrating visual pattern shifts.

3
00:00:28,500 --> 00:00:34,000
Notice how a naive thirty second clock cut chops this ongoing explanation right in half.

4
00:00:35,000 --> 00:00:41,200
This concludes the second module right before the forty two second visual transition.

5
00:00:43,000 --> 00:00:58,000
Finally we observe kinetic motion tracking in the third and final segment.
"""
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)


def test_cmasp_outperforms_fixed_window_on_synthetic_stream():
    """Validates boundary accuracy, zero discourse fracture (DBFI=0.0), and SCR/Q_chunk gains."""
    with tempfile.TemporaryDirectory() as tmpdir:
        v_path = os.path.join(tmpdir, "synthetic_test.mp4")
        s_path = os.path.join(tmpdir, "synthetic_test.srt")
        create_synthetic_test_video_and_srt(v_path, s_path)

        cfg = ChunkerConfig(
            min_chunk_sec=12.0,
            max_chunk_sec=30.0,
            target_chunk_sec=20.0,
            sample_fps=2.0,
        )
        chunker = IntelligentVideoChunker(cfg)
        result = chunker.chunk_video(v_path, srt_path=s_path)

        assert len(result.chunks) == 3, f"Expected 3 chunks, got {len(result.chunks)}"
        b1 = result.chunks[0].end_sec
        b2 = result.chunks[1].end_sec
        assert abs(b1 - 20.0) <= 1.5, f"Expected cut near 20.0s, got {b1}s"
        assert abs(b2 - 42.0) <= 1.5, f"Expected cut near 42.0s, got {b2}s"
        assert result.metrics["DBFI"] == 0.0, f"Expected DBFI=0.0, got {result.metrics['DBFI']}"

        timestamps, frames_bgr, duration_sec = chunker.visual_extractor.decode_sparse_frames(
            v_path
        )
        visual_flux, _ = chunker.visual_extractor.compute_visual_signals(frames_bgr)
        utterances = chunker.discourse_extractor.parse_srt(s_path)
        embeddings = chunker.semantic_extractor.compute_embeddings(frames_bgr)

        fixed_metrics = BaselineChunkers.fixed_window_chunker(
            v_path, 30.0, timestamps, embeddings, visual_flux, utterances, duration_sec, cfg
        )

        assert fixed_metrics["DBFI"] == 1.0, f"Expected Fixed-30s DBFI=1.0, got {fixed_metrics['DBFI']}"
        assert result.metrics["SCR"] > fixed_metrics["SCR"]
        assert result.metrics["Q_chunk"] > fixed_metrics["Q_chunk"]


def test_static_gate_surveillance_detection():
    """Tests that a completely static video feed is correctly flagged as STATIC_IDLE."""
    with tempfile.TemporaryDirectory() as tmpdir:
        v_path = os.path.join(tmpdir, "surveillance_test.mp4")
        width, height, fps = 64, 64, 5
        writer = _get_video_writer(v_path, float(fps), width, height)

        total_frames = 25 * fps
        frame = np.full((height, width, 3), 120, dtype=np.uint8)
        for _ in range(total_frames):
            writer.write(frame)
        writer.release()

        cfg = ChunkerConfig(
            min_chunk_sec=10.0,
            max_chunk_sec=30.0,
            target_chunk_sec=20.0,
            sample_fps=2.0,
            static_flux_threshold=0.015,
            min_static_duration_sec=15.0,
            suppress_static_segments=True,
        )
        chunker = IntelligentVideoChunker(cfg)
        result = chunker.chunk_video(v_path, srt_path=None)

        assert len(result.chunks) >= 1
        for c in result.chunks:
            if c.duration_sec >= 15.0:
                assert c.chunk_type == "STATIC_IDLE"
        assert result.detected_profile == "SURVEILLANCE_STATIC_SPARSE"


def test_discourse_srt_parser_edge_cases():
    """Tests SRT parsing resilience against empty strings, missing files, and unpunctuated lines."""
    extractor = DiscourseSignalExtractor()
    assert extractor.parse_srt("non_existent_file.srt") == []

    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
        f.write("")
        empty_srt_path = f.name
    try:
        assert extractor.parse_srt(empty_srt_path) == []
    finally:
        if os.path.exists(empty_srt_path):
            os.remove(empty_srt_path)

    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
        f.write("""1
00:00:01,000 --> 00:00:04,000
This is a full sentence.

2
00:00:05,000 --> 00:00:08,000
This is an incomplete clause without dot
""")
        punct_srt_path = f.name
    try:
        intervals = extractor.parse_srt(punct_srt_path)
        assert len(intervals) == 2
        assert intervals[0].ends_sentence is True
        assert intervals[1].ends_sentence is False
    finally:
        if os.path.exists(punct_srt_path):
            os.remove(punct_srt_path)


def test_youtube_rolling_caption_deoverlap_and_music_filter():
    """Validates V3 fix for overlapping 2-line YouTube auto-captions and [Music] tags."""
    extractor = DiscourseSignalExtractor(boundary_margin_sec=0.25)
    rolling_srt = """1
00:00:01,000 --> 00:00:05,000
[Music] [Applause]

2
00:00:05,000 --> 00:00:12,500
welcome everyone to our lecture on brain metabolism

3
00:00:09,800 --> 00:00:16,400
on brain metabolism and how exercise improves cognitive function.
"""
    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False) as f:
        f.write(rolling_srt)
        srt_path = f.name

    try:
        utterances = extractor.parse_srt(srt_path)
        # Cue 1 ([Music] [Applause]) must be filtered out; Cues 2 and 3 must be de-overlapped
        assert len(utterances) == 2
        assert utterances[0].end_sec <= utterances[1].start_sec
        assert utterances[0].end_sec == pytest.approx(9.8, abs=1e-2)
        # Transition at t=9.8s must be completely fracture-free
        assert extractor.is_timestamp_fractured(9.8, utterances) is False
        # Repeated rolling prefix "on brain metabolism" must be deduplicated from Cue 3
        assert not utterances[1].text.lower().startswith("on brain metabolism")
    finally:
        if os.path.exists(srt_path):
            os.remove(srt_path)


def test_subframe_discourse_boundary_snapping():
    """Verifies that coarse boundaries inside an utterance snap to exact fracture-free pauses."""
    extractor = DiscourseSignalExtractor(boundary_margin_sec=0.25)
    utterances = [
        UtteranceInterval(start_sec=10.0, end_sec=19.2, text="First spoken sentence."),
        UtteranceInterval(start_sec=19.6, end_sec=28.0, text="Second spoken sentence."),
    ]
    # Coarse grid cut at 20.5s fractures the second utterance [19.6, 28.0]
    assert extractor.is_timestamp_fractured(20.5, utterances) is True

    snapped = extractor.snap_boundary_to_discourse_pause(
        candidate_sec=20.5,
        prev_start_sec=0.0,
        next_end_sec=40.0,
        utterances=utterances,
        min_chunk_sec=10.0,
        max_chunk_sec=45.0,
        max_snap_sec=3.0,
    )
    # Snapped boundary must land in the safe pause around [19.2, 19.6] and be fracture-free
    assert extractor.is_timestamp_fractured(snapped, utterances) is False
    assert 19.15 <= snapped <= 19.65


def test_evaluator_metrics_edge_cases():
    """Verifies that ChunkingEvaluator gracefully handles empty chunk lists and small clips."""
    cfg = ChunkerConfig()
    empty_result = ChunkingEvaluator.evaluate(
        chunks=[],
        timestamps=np.array([]),
        embeddings=np.array([]),
        utterances=[],
        config=cfg,
        video_duration=0.0,
    )
    assert empty_result["SCR"] == 0.0
    assert empty_result["DBFI"] == 1.0
    assert empty_result["Q_chunk"] == 0.0


def test_dp_solver_prefix_variance_numerical_equivalence():
    """Validates that O(1) prefix-sum spherical variance is mathematically identical to O(N) calculation."""
    np.random.seed(42)
    raw = np.random.randn(50, 192).astype(np.float32)
    embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    prefix_emb = np.zeros((51, 192), dtype=np.float64)
    prefix_emb[1:] = np.cumsum(embeddings.astype(np.float64), axis=0)

    for i in [0, 5, 12]:
        for j in [i + 5, i + 15, i + 25]:
            count = j - i
            sum_vec = prefix_emb[j] - prefix_emb[i]
            mean_norm = float(np.linalg.norm(sum_vec)) / count
            o1_var = max(0.0, 1.0 - mean_norm)

            seg = embeddings[i:j]
            centroid = np.mean(seg, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            on_var = float(np.mean(1.0 - np.clip(seg @ centroid, -1.0, 1.0)))

            assert abs(o1_var - on_var) < 1e-5


def test_audio_rms_vad_synthetic_signal():
    """Tests that the Audio RMS VAD correctly extracts speech intervals from synthetic audio."""
    extractor = DiscourseSignalExtractor()
    sr = 16000
    duration_sec = 6.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    audio = np.random.normal(0, 0.005, len(t)).astype(np.float32)
    mask1 = (t >= 1.0) & (t <= 2.5)
    mask2 = (t >= 4.0) & (t <= 5.2)
    audio[mask1] += 0.5 * np.sin(2 * np.pi * 440.0 * t[mask1])
    audio[mask2] += 0.5 * np.sin(2 * np.pi * 440.0 * t[mask2])

    utterances = extractor.compute_audio_rms_vad(audio, sample_rate=sr)
    assert len(utterances) == 2
    assert abs(utterances[0].start_sec - 1.0) <= 0.25
    assert abs(utterances[0].end_sec - 2.5) <= 0.25
    assert abs(utterances[1].start_sec - 4.0) <= 0.25
    assert abs(utterances[1].end_sec - 5.2) <= 0.25


def test_semantic_extractor_properties():
    """Verifies that the fast spatial extractor outputs 192-D unit vectors even for pure black/white."""
    extractor = SemanticSignalExtractor(encoder_type="fast_spatial")

    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    textured = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    embs = extractor.compute_embeddings(np.stack([black, white, textured], axis=0))
    assert embs.shape == (3, 192)

    for i in range(3):
        norm = float(np.linalg.norm(embs[i]))
        assert abs(norm - 1.0) < 1e-4