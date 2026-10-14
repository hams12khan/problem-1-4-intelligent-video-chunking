"""Statistical Evaluation Suite: SCR, DBFI, IPU, Q_chunk, and CSR."""

from typing import Dict, List
import numpy as np
from src.chunker.config import ChunkerConfig, VideoChunk
from src.chunker.signals.discourse import UtteranceInterval


class ChunkingEvaluator:
    """Computes standard and custom information-theoretic chunking metrics."""

    @staticmethod
    def evaluate(
        chunks: List[VideoChunk],
        timestamps: np.ndarray,
        embeddings: np.ndarray,
        utterances: List[UtteranceInterval],
        config: ChunkerConfig,
        video_duration: float,
    ) -> Dict[str, float]:
        if not chunks:
            return {
                "SCR": 0.0,
                "DBFI": 1.0,
                "IPU": 0.0,
                "Q_chunk": 0.0,
                "CSR": 0.0,
                "mean_duration_sec": 0.0,
                "std_duration_sec": 0.0,
            }

        # 1. Constraint Satisfaction Rate (CSR)
        durations = np.array([c.duration_sec for c in chunks], dtype=np.float32)
        if video_duration < config.min_chunk_sec:
            csr = 1.0
        else:
            compliant = [
                (config.min_chunk_sec - 1.0) <= d <= (config.max_chunk_sec + 1.0)
                for d in durations
            ]
            csr = float(np.mean(compliant))

        # Map each chunk to its half-open frame indices [start_sec, end_sec)
        # so the transition frame at end_sec belongs cleanly to chunk k+1
        chunk_centroids = []
        intra_dispersions = []
        path_lengths = []
        num_chunks = len(chunks)

        for idx, c in enumerate(chunks):
            if idx < num_chunks - 1:
                mask = (timestamps >= c.start_sec - 1e-3) & (timestamps < c.end_sec - 1e-3)
            else:
                mask = (timestamps >= c.start_sec - 1e-3) & (timestamps <= c.end_sec + 1e-3)

            c_embs = embeddings[mask]
            if len(c_embs) == 0:
                nearest_idx = int(np.argmin(np.abs(timestamps - c.start_sec)))
                c_embs = embeddings[nearest_idx : nearest_idx + 1]

            # Spherical centroid
            raw_mean = np.mean(c_embs, axis=0)
            centroid = raw_mean / (np.linalg.norm(raw_mean) + 1e-8)
            chunk_centroids.append(centroid)

            # Intra-chunk cosine dispersion
            cos_sims = np.clip(c_embs @ centroid, -1.0, 1.0)
            intra_dispersions.append(float(np.mean(1.0 - cos_sims)))

            # Semantic path length V(C_k)
            if len(c_embs) > 1:
                step_diffs = np.linalg.norm(c_embs[1:] - c_embs[:-1], axis=1)
                path_lengths.append(float(np.sum(step_diffs)))
            else:
                path_lengths.append(0.0)

        # 2. Semantic Cohesion Ratio (SCR = Inter-Separation / Intra-Dispersion)
        mean_intra = float(np.mean(intra_dispersions))
        if len(chunk_centroids) > 1:
            centroids_mat = np.stack(chunk_centroids, axis=0)
            adj_dots = np.sum(centroids_mat[1:] * centroids_mat[:-1], axis=1)
            inter_sep = float(np.mean(np.clip(1.0 - adj_dots, 0.0, 2.0)))
        else:
            inter_sep = mean_intra + 0.05

        scr = inter_sep / (mean_intra + 1e-3)

        # 3. Discourse Boundary Fracture Index (DBFI)
        internal_boundaries = [c.end_sec for c in chunks[:-1]]
        if not internal_boundaries or not utterances:
            dbfi = 0.0
        else:
            fractures = 0
            margin = config.speech_boundary_margin_sec
            for b in internal_boundaries:
                fractured = any(
                    (u.start_sec + margin) < b < (u.end_sec - margin) for u in utterances
                )
                if fractured:
                    fractures += 1
            dbfi = float(fractures) / len(internal_boundaries)

        # 4. Information Path Uniformity (IPU)
        if len(path_lengths) > 1 and np.mean(path_lengths) > 1e-5:
            cv_path = float(np.std(path_lengths) / (np.mean(path_lengths) + 1e-8))
            ipu = 1.0 / (1.0 + cv_path)
        else:
            ipu = 1.0

        # 5. Composite Chunking Quality Score (Q_chunk)
        q_chunk = scr * (1.0 - dbfi) * ipu

        return {
            "SCR": float(scr),
            "DBFI": float(dbfi),
            "IPU": float(ipu),
            "Q_chunk": float(q_chunk),
            "CSR": float(csr),
            "mean_duration_sec": float(np.mean(durations)),
            "std_duration_sec": float(np.std(durations)),
        }