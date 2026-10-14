#!/usr/bin/env python3
"""Unified End-to-End Runner: Production Chunking + Scientific Benchmarking (V3).

Workflow:
1. PHASE 1: Runs C-MASP to chunk videos and generates production-ready JSON clips.
2. PHASE 2: Runs Baseline Comparators (Fixed-30s & Visual-Cut) and outputs the full
            scientific evaluation benchmark table (SCR, DBFI, IPU, Q_chunk, CSR, xRT).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.chunker.config import ChunkerConfig
from src.chunker.pipeline import BaselineChunkers, IntelligentVideoChunker


def find_matching_srt(video_path: str, subtitles_dir: Optional[Path] = None) -> str:
    """Finds matching SRT in the video folder or in subtitles_dir."""
    base_name = Path(video_path).stem
    for ext in [".srt", ".en.srt", ".vtt"]:
        candidate = Path(video_path).with_suffix(ext)
        if candidate.exists():
            return str(candidate)

    if subtitles_dir and subtitles_dir.exists():
        for srt_file in subtitles_dir.rglob("*.srt"):
            if srt_file.stem == base_name or srt_file.name.startswith(base_name):
                return str(srt_file)

    return ""


def run_pipeline(
    data_dir: str,
    max_videos: int = 0,
    encoder: str = "fast_spatial",
    device: str = "cuda",
    min_sec: float = 10.0,
    max_sec: float = 45.0,
    target_sec: float = 20.0,
    output_chunks: str = "production_chunks.json",
    output_benchmark: str = "benchmark_report.json",
):
    root = Path(data_dir).resolve()
    videos_dir = root / "videos" if (root / "videos").exists() else root
    subtitles_dir = root / "subtitles" if (root / "subtitles").exists() else root

    video_files = sorted(list(videos_dir.rglob("*.mp4")))
    if not video_files:
        print(f"[ERROR] No .mp4 videos found in {videos_dir}", file=sys.stderr)
        sys.exit(1)

    if max_videos > 0:
        video_files = video_files[:max_videos]

    config = ChunkerConfig(
        min_chunk_sec=min_sec,
        max_chunk_sec=max_sec,
        target_chunk_sec=target_sec,
        sample_fps=2.0,
        semantic_encoder=encoder,
        device=device,
    )
    chunker = IntelligentVideoChunker(config)

    if encoder == "open_clip":
        try:
            chunker.semantic_extractor._load_open_clip_if_needed()
        except Exception as e:
            print(f"[WARNING] Could not load open_clip ({e}); falling back to fast_spatial.")

    decode_sz = (
        config.clip_Input_size if encoder == "open_clip" else config.thumbnail_size
    )

    print("=" * 88)
    print("C-MASP: INTELLIGENT VIDEO CHUNKING & BENCHMARKING PIPELINE (V3)")
    print("=" * 88)
    print(f"Target Directory:    {root}")
    print(f"Total Videos:        {len(video_files)}")
    print(f"Encoder / Device:    {encoder} on {device} (decode_size={decode_sz}x{decode_sz})")
    print(f"Duration Bounds:     [{min_sec}s, {max_sec}s] (Target: {target_sec}s)")
    print(f"Output Chunks:       {output_chunks}")
    print(f"Output Benchmark:    {output_benchmark}")
    print("=" * 88)

    production_results = []
    benchmark_agg = {
        "Fixed_30s": {"SCR": [], "DBFI": [], "IPU": [], "Q_chunk": [], "CSR": []},
        "Visual_Cut_Only": {"SCR": [], "DBFI": [], "IPU": [], "Q_chunk": [], "CSR": []},
        "C_MASP_Ours": {
            "SCR": [],
            "DBFI": [],
            "IPU": [],
            "Q_chunk": [],
            "CSR": [],
            "xRT": [],
        },
    }

    t_start_all = time.perf_counter()

    for v_path in tqdm(video_files, desc="Processing Videos"):
        v_str = str(v_path)
        s_str = find_matching_srt(v_str, subtitles_dir)

        # 1. Single-Pass Signal Extraction
        t0 = time.perf_counter()
        timestamps, frames_bgr, duration_sec = (
            chunker.visual_extractor.decode_sparse_frames(v_str, decode_size=decode_sz)
        )
        visual_flux, visual_zscore = chunker.visual_extractor.compute_visual_signals(
            frames_bgr
        )
        utterances = chunker.discourse_extractor.parse_srt(s_str if s_str else None)
        if not utterances and config.use_audio_rms_fallback:
            utterances = chunker.discourse_extractor.extract_utterances_from_audio(v_str)

        speech_penalty = chunker.discourse_extractor.compute_speech_penalty_grid(
            timestamps, utterances
        )
        embeddings = chunker.semantic_extractor.compute_embeddings(frames_bgr)
        semantic_novelty = chunker.semantic_extractor.compute_foote_novelty(embeddings)

        # 2. PHASE 1: C-MASP Production Chunking
        res_cmasp = chunker.chunk_from_signals(
            video_path=v_str,
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
        production_results.append(res_cmasp.to_dict())

        # 3. PHASE 2: Scientific Baseline Comparators
        m_fixed = BaselineChunkers.fixed_window_chunker(
            v_str,
            30.0,
            timestamps,
            embeddings,
            visual_flux,
            utterances,
            duration_sec,
            config,
        )
        m_visual = BaselineChunkers.visual_cut_only_chunker(
            timestamps, embeddings, visual_zscore, utterances, duration_sec, config
        )

        for k in ["SCR", "DBFI", "IPU", "Q_chunk", "CSR"]:
            benchmark_agg["Fixed_30s"][k].append(m_fixed[k])
            benchmark_agg["Visual_Cut_Only"][k].append(m_visual[k])
            benchmark_agg["C_MASP_Ours"][k].append(res_cmasp.metrics[k])
        benchmark_agg["C_MASP_Ours"]["xRT"].append(res_cmasp.realtime_factor_xrt)

    total_wall_time = time.perf_counter() - t_start_all

    # 4. Save Production Chunks
    with open(output_chunks, "w", encoding="utf-8") as f:
        json.dump({"videos": production_results}, f, indent=2)

    # 5. Compute Benchmark Summary Table
    summary = {}
    for method, m_dict in benchmark_agg.items():
        summary[method] = {k: float(np.mean(vals)) for k, vals in m_dict.items() if vals}

    with open(output_benchmark, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_video": production_results}, f, indent=2)

    print("\n" + "=" * 88)
    print("PHASE 1 COMPLETE: PRODUCTION CHUNKS SAVED")
    print("=" * 88)
    print(f"Generated chunks for {len(production_results)} videos -> {output_chunks}")

    print("\n" + "=" * 88)
    print("PHASE 2 COMPLETE: SCIENTIFIC BENCHMARK & ABLATION RESULTS")
    print("=" * 88)
    print(
        f"{'Method':<22} | {'SCR (higher)':<12} | {'DBFI (lower)':<12} | {'IPU (higher)':<12} | "
        f"{'Q_chunk (higher)':<17} | {'CSR (higher)':<12}"
    )
    print("-" * 88)
    for method in ["Fixed_30s", "Visual_Cut_Only", "C_MASP_Ours"]:
        s = summary[method]
        print(
            f"{method:<22} | {s['SCR']:<12.4f} | {s['DBFI']:<12.4f} | {s['IPU']:<12.4f} | "
            f"{s['Q_chunk']:<17.4f} | {s['CSR']:<12.4f}"
        )
    print("=" * 88)
    mean_xrt = summary["C_MASP_Ours"].get("xRT", 0.0)
    print(
        f"Throughput Speed:   {mean_xrt:.1f}x Realtime ({mean_xrt * 30.0:.0f} effective FPS at 30fps native)"
    )
    print(f"Total Wall Clock:   {total_wall_time:.1f} seconds")
    print(f"Benchmark Report:   {output_benchmark}")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run C-MASP Video Chunking and Benchmark Evaluation in one script."
    )
    parser.add_argument(
        "--data-dir",
        default="../data",
        help="Directory containing videos/ and subtitles/.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Number of videos to process (0 = all).",
    )
    parser.add_argument(
        "--encoder",
        choices=["fast_spatial", "open_clip"],
        default="fast_spatial",
        help="Feature encoder.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device for neural models: cuda or cpu."
    )
    parser.add_argument(
        "--min-sec", type=float, default=10.0, help="Min chunk length (default: 10s)."
    )
    parser.add_argument(
        "--max-sec", type=float, default=45.0, help="Max chunk length (default: 45s)."
    )
    parser.add_argument(
        "--target-sec",
        type=float,
        default=20.0,
        help="Target chunk length (default: 20s).",
    )
    parser.add_argument(
        "--output-chunks",
        default="production_chunks.json",
        help="Path to save chunk JSON.",
    )
    parser.add_argument(
        "--output-benchmark",
        default="benchmark_report.json",
        help="Path to save benchmark JSON.",
    )
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        max_videos=args.max_videos,
        encoder=args.encoder,
        device=args.device,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        target_sec=args.target_sec,
        output_chunks=args.output_chunks,
        output_benchmark=args.output_benchmark,
    )