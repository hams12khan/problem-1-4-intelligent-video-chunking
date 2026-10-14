"""Tier 4: Linear-Time Constrained Dynamic Programming (Viterbi) Boundary Optimizer."""

from typing import List, Tuple
import numpy as np
from src.chunker.config import ChunkerConfig


class ConstrainedDPBoundarySolver:
    """Finds the globally optimal partition of a video timeline in O(T * W_max) time."""

    def __init__(self, config: ChunkerConfig):
        self.cfg = config

    def solve(
        self,
        timestamps: np.ndarray,
        embeddings: np.ndarray,
        visual_zscore: np.ndarray,
        semantic_novelty: np.ndarray,
        speech_penalty: np.ndarray,
        video_duration: float,
    ) -> List[Tuple[int, int, str]]:
        """Solves for optimal half-open frame-index intervals [(start_idx, end_idx, reason), ...].

        Principal Scientist V3 Formulation:
        1. Duration-Weighted Half-Open Spherical Variance on [i, j) (eliminates chunk-count bias).
        2. Exact Inter-Chunk Centroid Contrast against predecessor chunk [backpointer[i], i) AND lookahead.
        3. Information Path Uniformity (Entropy Equipartition) on cumulative step variations.
        4. Internal Snapping Cushion (+-1.25s on long streams) so post-DP sub-frame speech snapping
           always has bidirectional slack without violating [min_chunk_sec, max_chunk_sec].
        """
        n = len(timestamps)
        if n <= 2:
            return [(0, max(0, n - 1), "SINGLE_SHORT_CLIP")]
        if video_duration <= self.cfg.min_chunk_sec * 1.5:
            return [(0, n - 1, "SINGLE_SHORT_CLIP")]

        fps = self.cfg.sample_fps
        # Reserve a 1.25s inner cushion on long videos so continuous speech snapping always has slack
        if (
            video_duration > self.cfg.max_chunk_sec * 1.5
            and (self.cfg.max_chunk_sec - self.cfg.min_chunk_sec) >= 10.0
        ):
            eff_min_sec = self.cfg.min_chunk_sec + 1.25
            eff_max_sec = self.cfg.max_chunk_sec - 1.25
        else:
            eff_min_sec = self.cfg.min_chunk_sec
            eff_max_sec = self.cfg.max_chunk_sec

        min_steps = max(1, int(round(eff_min_sec * fps)))
        max_steps = max(min_steps + 1, int(round(eff_max_sec * fps)))

        # 1. Precompute prefix sums of embeddings for O(1) spherical variance & centroid lookup
        emb_dim = embeddings.shape[1]
        prefix_emb = np.zeros((n + 1, emb_dim), dtype=np.float64)
        prefix_emb[1:] = np.cumsum(embeddings.astype(np.float64), axis=0)

        # 2. Precompute prefix sums of frame-to-frame semantic path variations for O(1) IPU optimization
        prefix_path = np.zeros(n, dtype=np.float64)
        if n > 1:
            step_diffs = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
            prefix_path[1:] = np.cumsum(step_diffs.astype(np.float64))

        total_path = float(prefix_path[-1])
        est_num_chunks = max(1.0, video_duration / self.cfg.target_chunk_sec)
        target_chunk_path = total_path / est_num_chunks

        # 3. Cross-modal J-cut / L-cut receptive field smoothing (shares visual/semantic bonus within +-1s)
        smooth_vz = np.copy(visual_zscore)
        smooth_sn = np.copy(semantic_novelty)
        if n >= 5:
            kernel_jl = np.array([0.30, 0.65, 1.0, 0.65, 0.30], dtype=np.float32)
            for t in range(n):
                lo = max(0, t - 2)
                hi = min(n, t + 3)
                k_slice = kernel_jl[2 - (t - lo) : 2 + (hi - t)]
                smooth_vz[t] = float(np.max(visual_zscore[lo:hi] * k_slice))
                smooth_sn[t] = float(np.max(semantic_novelty[lo:hi] * k_slice))

        # 4. Precompute O(1) Forward Lookahead Centroids at each potential boundary j
        look_w = max(2, min_steps // 2)
        forward_centroids = np.zeros((n, emb_dim), dtype=np.float64)
        for j in range(n):
            f_end = min(n, j + look_w)
            f_sum = prefix_emb[f_end] - prefix_emb[j]
            f_norm = float(np.linalg.norm(f_sum))
            if f_norm > 1e-8:
                forward_centroids[j] = f_sum / f_norm
            else:
                forward_centroids[j] = embeddings[j]

        def half_open_stats(idx_start: int, idx_end: int) -> Tuple[float, np.ndarray]:
            """Computes spherical variance and unit centroid on half-open [idx_start, idx_end)."""
            count = idx_end - idx_start
            if count <= 1:
                return 0.0, embeddings[idx_start].astype(np.float64)
            sum_vec = prefix_emb[idx_end] - prefix_emb[idx_start]
            vec_norm = float(np.linalg.norm(sum_vec))
            mean_norm = vec_norm / count
            var_val = max(0.0, 1.0 - mean_norm)
            centroid = sum_vec / (vec_norm + 1e-8)
            return var_val, centroid

        # Dynamic Programming tables + optimal predecessor chunk centroids
        dp_cost = np.full(n, np.inf, dtype=np.float64)
        backpointer = np.zeros(n, dtype=np.int32)
        best_prev_centroid = np.zeros((n, emb_dim), dtype=np.float64)
        dp_cost[0] = 0.0

        for j in range(min_steps, n):
            i_min = max(0, j - max_steps)
            i_max = j - min_steps

            if j < n - 1:
                cut_cost = (
                    self.cfg.weight_speech_interruption_penalty * float(speech_penalty[j])
                    - self.cfg.weight_visual_cut_bonus * float(smooth_vz[j])
                    - self.cfg.weight_semantic_novelty_bonus * float(smooth_sn[j])
                )
            else:
                cut_cost = 0.0

            best_val = np.inf
            best_i = i_min
            best_c = embeddings[j].astype(np.float64)
            fwd_c = forward_centroids[j]

            for i in range(i_min, i_max + 1):
                if np.isinf(dp_cost[i]):
                    continue

                dur = float(timestamps[j] - timestamps[i])
                dur_scale = dur / self.cfg.target_chunk_sec
                len_dev = (dur_scale - 1.0) ** 2

                # Half-open interval [i, j) excludes transition frame j from preceding chunk
                end_excl = j if j < n - 1 else n
                var_cost, seg_centroid = half_open_stats(i, end_excl)

                # Duration-weighted intra-chunk dispersion (Ward's minimum variance principle)
                weighted_var = dur_scale * var_cost

                # Exact inter-chunk contrast against predecessor chunk [backpointer[i], i) + lookahead
                if j < n - 1:
                    fwd_contrast = max(0.0, 1.0 - float(np.dot(seg_centroid, fwd_c)))
                else:
                    fwd_contrast = 0.0

                if i > 0:
                    prev_contrast = max(
                        0.0, 1.0 - float(np.dot(seg_centroid, best_prev_centroid[i]))
                    )
                    inter_cos_dist = (
                        0.55 * prev_contrast + 0.45 * fwd_contrast
                        if j < n - 1
                        else prev_contrast
                    )
                else:
                    inter_cos_dist = fwd_contrast

                # Information Path Uniformity (IPU) equipartition term on [i, end_excl)
                if target_chunk_path > 1e-4 and end_excl > i + 1:
                    seg_path = float(prefix_path[end_excl - 1] - prefix_path[i])
                    path_dev = ((seg_path - target_chunk_path) / (target_chunk_path + 1e-4)) ** 2
                    path_cost = min(path_dev, 2.0)
                else:
                    path_cost = 0.0

                segment_cost = (
                    self.cfg.weight_intra_variance * weighted_var
                    - self.cfg.weight_inter_contrast_bonus * inter_cos_dist
                    + self.cfg.weight_path_uniformity * path_cost
                    + self.cfg.weight_length_prior * len_dev
                    + cut_cost
                )
                candidate_total = dp_cost[i] + segment_cost
                if candidate_total < best_val:
                    best_val = candidate_total
                    best_i = i
                    best_c = seg_centroid

            dp_cost[j] = best_val
            backpointer[j] = best_i
            best_prev_centroid[j] = best_c

        # Backtrack from final state (or nearest valid tail)
        end_cursor = n - 1
        if np.isinf(dp_cost[end_cursor]):
            valid_indices = np.where(np.isfinite(dp_cost))[0]
            if len(valid_indices) <= 1:
                return [(0, n - 1, "FALLBACK_FULL_VIDEO")]
            end_cursor = int(valid_indices[-1])

        if end_cursor == n - 1:
            boundaries = [n - 1]
            curr = n - 1
            while curr > 0:
                prev = int(backpointer[curr])
                boundaries.append(prev)
                if prev <= 0:
                    break
                curr = prev
        else:
            tail_frames = (n - 1) - end_cursor
            hard_min_steps = max(1, int(round(self.cfg.min_chunk_sec * fps)))
            if tail_frames >= hard_min_steps:
                boundaries = [n - 1, end_cursor]
            else:
                boundaries = [n - 1]
            curr = end_cursor
            while curr > 0:
                prev = int(backpointer[curr])
                boundaries.append(prev)
                if prev <= 0:
                    break
                curr = prev

        boundaries = sorted(list(set(boundaries)))
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        if boundaries[-1] != n - 1:
            boundaries[-1] = n - 1

        intervals: List[Tuple[int, int, str]] = []
        for k in range(len(boundaries) - 1):
            s_idx = boundaries[k]
            e_idx = boundaries[k + 1]
            reason = self._classify_boundary_reason(
                e_idx, n, visual_zscore, semantic_novelty, speech_penalty
            )
            intervals.append((s_idx, e_idx, reason))

        return intervals

    @staticmethod
    def _classify_boundary_reason(
        idx: int,
        n: int,
        visual_zscore: np.ndarray,
        semantic_novelty: np.ndarray,
        speech_penalty: np.ndarray,
    ) -> str:
        if idx >= n - 1:
            return "END_OF_STREAM"
        lo = max(0, idx - 2)
        hi = min(n, idx + 3)
        vz = float(np.max(visual_zscore[lo:hi]))
        sn = float(np.max(semantic_novelty[lo:hi]))
        sp = float(speech_penalty[idx])

        if vz >= 0.55 and sp == 0.0:
            return "VISUAL_SHOT_CUT_SPEECH_ALIGNED"
        if vz >= 0.55:
            return "VISUAL_SHOT_CUT"
        if sn >= 0.50 and sp == 0.0:
            return "SEMANTIC_SHIFT_SPEECH_PAUSE"
        if sn >= 0.50:
            return "SEMANTIC_REGIME_SHIFT"
        if sp == 0.0:
            return "DISCOURSE_SENTENCE_BOUNDARY"
        return "DURATION_WINDOW_OPTIMAL"
