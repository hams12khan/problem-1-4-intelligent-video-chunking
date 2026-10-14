"""Main Orchestrator for C-MASP Intelligent Video Chunker and Baseline Comparators."""

import time
from typing import Dict, List, Optional
import numpy as np

from src.chunker.config import ChunkerConfig, ChunkingResult, VideoChunk
from src.chunker.evaluation.metrics import ChunkingEvaluator
from src.chunker.optimizer.dp_solver import ConstrainedDPBoundarySolver
from src.chunker.signals.discourse import DiscourseSignalExtractor, UtteranceInterval
from src.chunker.signals.semantic import SemanticSignalExtractor
from src.chunker.signals.visual import VisualSignalExtractor


class IntelligentVideoChunker:
    """Cascaded Multi-Modal Adaptive Semantic Partitioning (C-MASP) Engine."""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self.visual_extractor = VisualSignalExtractor(
            sample_fps=self.config.sample_fps,
            thumbnail_size=self.config.thumbnail_size,
            zscore_window_sec=self.config.zscore_window_sec,
        )
        self.discourse_extractor = DiscourseSignalExtractor(
            boundary_margin_sec=self.config.speech_boundary_margin_sec
        )
        self.semantic_extractor = SemanticSignalExtractor(
            encoder_type=self.config.semantic_encoder,
            clip_model_name=self.config.clip_model_name,
            clip_pretrained=self.config.clip_pretrained,
            foote_half_width_sec=self.config.foote_kernel_half_width_sec,
            sample_fps=self.config.sample_fps,
            device=self.config.device,
        )
        self.dp_solver = ConstrainedDPBoundarySolver(self.config)

    @staticmethod
    def _detect_video_profile(
        visual_flux: np.ndarray,
        visual_zscore: np.ndarray,
        utterances: List[UtteranceInterval],
        duration_sec: float,
    ) -> str:
        """Classifies the video structural regime using median motion vs. discrete cut rates."""
        mean_flux = float(np.mean(visual_flux))
        median_flux = float(np.median(visual_flux))
        static_ratio = float(np.mean(visual_flux < 0.015))
        speech_coverage = (
            sum(max(0.0, u.end_sec - u.start_sec) for u in utterances)
            / max(duration_sec, 1.0)
            if utterances
            else 0.0
        )
        true_cuts = np.sum((visual_zscore >= 0.55) & (visual_flux >= 0.14))
        cut_rate_per_min = float(true_cuts) / max(duration_sec / 60.0, 0.1)

        if static_ratio > 0.70 and speech_coverage < 0.15:
            return "SURVEILLANCE_STATIC_SPARSE"
        if speech_coverage > 0.40 and mean_flux < 0.09:
            return "MEETING_LECTURE_DIALOGUE"
        # Continuous high-motion sports/gameplay has sustained high median frame-to-frame flux
        if median_flux > 0.24 and mean_flux > 0.24:
            return "SPORTS_GAMEPLAY_HIGH_MOTION"
        if cut_rate_per_min >= 8.5 or (mean_flux > 0.18 and speech_coverage < 0.35):
            return "CINEMATIC_RAPID_MONTAGE"
        return "GENERAL_MULTIMODAL_BROADCAST"

    def chunk_from_signals(
        self,
        video_path: str,
        duration_sec: float,
        timestamps: np.ndarray,
        visual_flux: np.ndarray,
        visual_zscore: np.ndarray,
        utterances: List[UtteranceInterval],
        embeddings: np.ndarray,
        semantic_novelty: np.ndarray,
        speech_penalty: np.ndarray,
        start_time: Optional[float] = None,
    ) -> ChunkingResult:
        """Executes Tier 4 DP optimization, bidirectional speech snapping, and telemetry generation."""
        t0 = start_time if start_time is not None else time.perf_counter()

        profile = self._detect_video_profile(
            visual_flux, visual_zscore, utterances, duration_sec
        )

        # Tier 4: Constrained Dynamic Programming boundary solver on coarse frame grid
        raw_intervals = self.dp_solver.solve(
            timestamps=timestamps,
            embeddings=embeddings,
            visual_zscore=visual_zscore,
            semantic_novelty=semantic_novelty,
            speech_penalty=speech_penalty,
            video_duration=duration_sec,
        )

        # Extract coarse boundary timestamps [0.0, b_1, b_2, ..., duration_sec]
        num_segs = len(raw_intervals)
        boundary_times = [0.0]
        for idx in range(num_segs - 1):
            e_i = raw_intervals[idx][1]
            boundary_times.append(float(timestamps[e_i]))
        boundary_times.append(float(duration_sec))

        # Two-Pass Forward-Backward Sub-Frame Linguistic Snapping
        if utterances and num_segs > 1:
            for _ in range(2):
                for k in range(1, num_segs):
                    snapped_t = self.discourse_extractor.snap_boundary_to_discourse_pause(
                        candidate_sec=boundary_times[k],
                        prev_start_sec=boundary_times[k - 1],
                        next_end_sec=boundary_times[k + 1],
                        utterances=utterances,
                        min_chunk_sec=self.config.min_chunk_sec,
                        max_chunk_sec=self.config.max_chunk_sec,
                        max_snap_sec=self.config.speech_snap_window_sec,
                    )
                    boundary_times[k] = snapped_t

        chunks: List[VideoChunk] = []
        for idx, (s_i, e_i, reason) in enumerate(raw_intervals):
            start_s = boundary_times[idx]
            end_s = boundary_times[idx + 1]
            if end_s <= start_s:
                continue

            # Map snapped continuous boundaries back to half-open frame slice [start_s, end_s)
            if idx < num_segs - 1:
                mask = (timestamps >= start_s - 1e-3) & (timestamps < end_s - 1e-3)
            else:
                mask = (timestamps >= start_s - 1e-3) & (timestamps <= end_s + 1e-3)

            if np.any(mask):
                seg_flux = float(np.mean(visual_flux[mask]))
                seg_embs = embeddings[mask]
            else:
                end_slice = e_i if (idx < num_segs - 1 and e_i > s_i) else e_i + 1
                seg_flux = float(np.mean(visual_flux[s_i:end_slice]))
                seg_embs = embeddings[s_i:end_slice]

            centroid = np.mean(seg_embs, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            intra_var = float(np.mean(1.0 - np.clip(seg_embs @ centroid, -1.0, 1.0)))

            is_static = (
                self.config.suppress_static_segments
                and seg_flux < self.config.static_flux_threshold
                and (end_s - start_s) >= self.config.min_static_duration_sec
                and not self.discourse_extractor.extract_chunk_transcript(
                    start_s, end_s, utterances
                )
            )
            c_type = "STATIC_IDLE" if is_static else "ACTIVE_CONTENT"

            speech_ok = (
                not self.discourse_extractor.is_timestamp_fractured(end_s, utterances)
                if idx < num_segs - 1
                else True
            )
            if speech_ok and reason == "VISUAL_SHOT_CUT":
                reason = "VISUAL_SHOT_CUT_SPEECH_ALIGNED"
            elif speech_ok and reason == "SEMANTIC_REGIME_SHIFT":
                reason = "SEMANTIC_SHIFT_SPEECH_PAUSE"
            elif speech_ok and reason == "DURATION_WINDOW_OPTIMAL" and utterances:
                reason = "DISCOURSE_SENTENCE_BOUNDARY"

            snippet = self.discourse_extractor.extract_chunk_transcript(
                start_s, end_s, utterances
            )

            chunks.append(
                VideoChunk(
                    chunk_id=idx,
                    start_sec=start_s,
                    end_sec=end_s,
                    duration_sec=end_s - start_s,
                    chunk_type=c_type,
                    boundary_reason=reason,
                    intra_semantic_variance=intra_var,
                    visual_flux_mean=seg_flux,
                    speech_aligned=speech_ok,
                    transcript_snippet=snippet,
                )
            )

        elapsed = max(time.perf_counter() - t0, 1e-6)
        xrt = duration_sec / elapsed

        metrics = ChunkingEvaluator.evaluate(
            chunks=chunks,
            timestamps=timestamps,
            embeddings=embeddings,
            utterances=utterances,
            config=self.config,
            video_duration=duration_sec,
        )

        return ChunkingResult(
            video_path=video_path,
            video_duration_sec=duration_sec,
            detected_profile=profile,
            processing_time_sec=elapsed,
            realtime_factor_xrt=xrt,
            chunks=chunks,
            metrics=metrics,
        )

    def chunk_video(
        self, video_path: str, srt_path: Optional[str] = None
    ) -> ChunkingResult:
        """Executes the full 4-Tier C-MASP chunking pipeline on a video file."""
        t0 = time.perf_counter()

        decode_sz = (
            self.config.clip_Input_size
            if self.config.semantic_encoder == "open_clip"
            else self.config.thumbnail_size
        )
        timestamps, frames_bgr, duration_sec = self.visual_extractor.decode_sparse_frames(
            video_path, decode_size=decode_sz
        )
        visual_flux, visual_zscore = self.visual_extractor.compute_visual_signals(frames_bgr)

        utterances = self.discourse_extractor.parse_srt(srt_path)
        if not utterances and self.config.use_audio_rms_fallback:
            utterances = self.discourse_extractor.extract_utterances_from_audio(video_path)

        speech_penalty = self.discourse_extractor.compute_speech_penalty_grid(
            timestamps, utterances
        )

        embeddings = self.semantic_extractor.compute_embeddings(frames_bgr)
        semantic_novelty = self.semantic_extractor.compute_foote_novelty(embeddings)

        return self.chunk_from_signals(
            video_path=video_path,
            duration_sec=duration_sec,
            timestamps=timestamps,
            visual_flux=visual_flux,
            visual_zscore=visual_zscore,
            utterances=utterances,
            embeddings=embeddings,
            semantic_novelty=semantic_novelty,
            speech_penalty=speech_penalty,
            start_time=t0,
        )


class BaselineChunkers:
    """Implements Fixed-Time and Visual-Only baselines for rigorous ablation benchmarking."""

    @staticmethod
    def fixed_window_chunker(
        video_path: str,
        window_sec: float,
        timestamps: np.ndarray,
        embeddings: np.ndarray,
        visual_flux: np.ndarray,
        utterances: List[UtteranceInterval],
        duration_sec: float,
        config: ChunkerConfig,
    ) -> Dict[str, float]:
        chunks: List[VideoChunk] = []
        curr = 0.0
        idx = 0
        while curr < duration_sec - 1.0:
            nxt = min(duration_sec, curr + window_sec)
            chunks.append(
                VideoChunk(
                    chunk_id=idx,
                    start_sec=curr,
                    end_sec=nxt,
                    duration_sec=nxt - curr,
                    chunk_type="ACTIVE_CONTENT",
                    boundary_reason="FIXED_CLOCK_CUT",
                    intra_semantic_variance=0.0,
                    visual_flux_mean=0.0,
                    speech_aligned=False,
                )
            )
            curr = nxt
            idx += 1

        if chunks and chunks[-1].end_sec < duration_sec:
            chunks[-1].end_sec = duration_sec
            chunks[-1].duration_sec = duration_sec - chunks[-1].start_sec

        return ChunkingEvaluator.evaluate(
            chunks, timestamps, embeddings, utterances, config, duration_sec
        )

    @staticmethod
    def visual_cut_only_chunker(
        timestamps: np.ndarray,
        embeddings: np.ndarray,
        visual_zscore: np.ndarray,
        utterances: List[UtteranceInterval],
        duration_sec: float,
        config: ChunkerConfig,
    ) -> Dict[str, float]:
        """Greedy visual-cut detector that cuts whenever visual_zscore > 0.55."""
        cut_indices = [0]
        last_t = 0.0
        for i, (t, z) in enumerate(zip(timestamps, visual_zscore)):
            if z > 0.55 and (t - last_t) >= config.min_chunk_sec:
                cut_indices.append(i)
                last_t = float(t)
            elif (t - last_t) >= config.max_chunk_sec:
                cut_indices.append(i)
                last_t = float(t)

        if cut_indices[-1] != len(timestamps) - 1:
            cut_indices.append(len(timestamps) - 1)

        chunks: List[VideoChunk] = []
        for idx in range(len(cut_indices) - 1):
            s_t = 0.0 if idx == 0 else float(timestamps[cut_indices[idx]])
            e_t = (
                duration_sec
                if idx == len(cut_indices) - 2
                else float(timestamps[cut_indices[idx + 1]])
            )
            if e_t > s_t:
                chunks.append(
                    VideoChunk(
                        chunk_id=idx,
                        start_sec=s_t,
                        end_sec=e_t,
                        duration_sec=e_t - s_t,
                        chunk_type="ACTIVE_CONTENT",
                        boundary_reason="GREEDY_VISUAL_CUT",
                        intra_semantic_variance=0.0,
                        visual_flux_mean=0.0,
                        speech_aligned=False,
                    )
                )

        return ChunkingEvaluator.evaluate(
            chunks, timestamps, embeddings, utterances, config, duration_sec
        )
