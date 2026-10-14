#!/usr/bin/env python3
"""Automated Benchmark & Ablation Suite on aintropy-ai/video-mme-sample10."""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.chunker.config import ChunkerConfig
from src.chunker.pipeline import BaselineChunkers, IntelligentVideoChunker


def prepare_videomme_sample10(data_dir: str, max_videos: int = 0) -> List[Dict[str, str]]:
    """Downloads or locates the aintropy-ai/video-mme-sample10 dataset and matches videos with SRTs."""
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(root.rglob("*.mp4")))
    if not video_files:
        print(f"[1/3] Syncing dataset 'aintropy-ai/video-mme-sample10' into {root}...")
        try:
            from huggingface_hub import hf_hub_download, snapshot_download

            print("Downloading subtitles and metadata (~3 MB)...")
            snapshot_download(
                repo_id="aintropy-ai/video-mme-sample10",
                repo_type="dataset",
                local_dir=str(root),
                allow_patterns=["manifest.json", "sample_ids.txt", "qa/*", "subtitles/*"],
            )

            sample_ids_path = root / "sample_ids.txt"
            video_ids = []
            if sample_ids_path.exists():
                with open(sample_ids_path, "r", encoding="utf-8") as f:
                    video_ids = [line.strip() for line in f if line.strip()]

            target_ids = video_ids[:max_videos] if max_videos > 0 else video_ids

            print(f"Downloading {len(target_ids)} target videos from Hugging Face...")
            for idx, vid in enumerate(target_ids, 1):
                v_filename = f"videos/{vid}.mp4"
                local_path = root / v_filename
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue
                print(f"  [{idx}/{len(target_ids)}] Downloading {vid}.mp4...")
                hf_hub_download(
                    repo_id="aintropy-ai/video-mme-sample10",
                    repo_type="dataset",
                    filename=v_filename,
                    local_dir=str(root),
                )
        except Exception as e:
            print(f"[WARNING] HuggingFace download notice: {e}. Checking existing local files.")

    for archive_name in ["videos.zip", "subtitles.zip"]:
        zip_path = root / archive_name
        if zip_path.exists():
            extract_marker = root / f".extracted_{archive_name}"
            if not extract_marker.exists():
                print(f"Extracting {archive_name}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(root)
                extract_marker.touch()

    video_files = sorted(list(root.rglob("*.mp4")))
    srt_map = {p.stem: str(p) for p in root.rglob("*.srt")}

    dataset_items = []
    for vf in video_files:
        dataset_items.append(
            {
                "video_id": vf.stem,
                "video_path": str(vf),
                "srt_path": srt_map.get(vf.stem, ""),
            }
        )

    print(f"[2/3] Found {len(dataset_items)} local video files with {len(srt_map)} subtitle files.")
    return dataset_items


def run_benchmark(data_dir: str, max_videos: int, output_json: str, encoder: str = "fast_spatial", device: str = "cpu"):
    items = prepare_videomme_sample10(data_dir, max_videos)
    if not items:
        print(f"[ERROR] No video files found in {data_dir} to benchmark.")
        return

    if max_videos > 0:
        items = items[:max_videos]

    config = ChunkerConfig(
        min_chunk_sec=10.0,
        max_chunk_sec=45.0,
        target_chunk_sec=25.0,
        sample_fps=2.0,
        semantic_encoder=encoder,
        device=device,
    )
    chunker = IntelligentVideoChunker(config)

    agg = {
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

    per_video_records = []
    print(f"[3/3] Running single-pass 3-way ablation benchmark across {len(items)} videos with encoder='{encoder}' on device='{device}'...")

    for item in tqdm(items, desc="Benchmarking VideoMME"):
        v_path = item["video_path"]
        s_path = item["srt_path"]

        # Single-pass signal extraction (eliminates redundant double video decoding)
        t0 = time.perf_counter()
        timestamps, frames_bgr, duration_sec = chunker.visual_extractor.decode_sparse_frames(
            v_path
        )
        visual_flux, visual_zscore = chunker.visual_extractor.compute_visual_signals(frames_bgr)
        utterances = chunker.discourse_extractor.parse_srt(s_path if s_path else None)
        if not utterances and config.use_audio_rms_fallback:
            utterances = chunker.discourse_extractor.extract_utterances_from_audio(v_path)
        speech_penalty = chunker.discourse_extractor.compute_speech_penalty_grid(
            timestamps, utterances
        )
        embeddings = chunker.semantic_extractor.compute_embeddings(frames_bgr)
        semantic_novelty = chunker.semantic_extractor.compute_foote_novelty(embeddings)

        # Run full C-MASP pipeline from signals (includes t0 start_time for accurate end-to-end xRT)
        res_ours = chunker.chunk_from_signals(
            video_path=v_path,
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

        # Run Baseline 1: Naive Fixed 30s Window
        m_fixed = BaselineChunkers.fixed_window_chunker(
            v_path, 30.0, timestamps, embeddings, visual_flux, utterances, duration_sec, config
        )

        # Run Baseline 2: Greedy Visual Cut Only
        m_visual = BaselineChunkers.visual_cut_only_chunker(
            timestamps, embeddings, visual_zscore, utterances, duration_sec, config
        )

        for k in ["SCR", "DBFI", "IPU", "Q_chunk", "CSR"]:
            agg["Fixed_30s"][k].append(m_fixed[k])
            agg["Visual_Cut_Only"][k].append(m_visual[k])
            agg["C_MASP_Ours"][k].append(res_ours.metrics[k])
        agg["C_MASP_Ours"]["xRT"].append(res_ours.realtime_factor_xrt)

        per_video_records.append(res_ours.to_dict())

    summary = {}
    for method, m_dict in agg.items():
        summary[method] = {k: float(np.mean(vals)) for k, vals in m_dict.items() if vals}

    print("\n" + "=" * 88)
    print(f"VIDEOMME-SAMPLE10 ABLATION BENCHMARK RESULTS (Encoder: {encoder}, Device: {device})")
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
    print(
        f"Mean C-MASP Throughput: {summary['C_MASP_Ours'].get('xRT', 0.0):.1f}x Realtime "
        f"({summary['C_MASP_Ours'].get('xRT', 0.0) * 30.0:.0f} effective FPS at 30fps native)"
    )

    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "videos": per_video_records}, f, indent=2)
    print(f"Full benchmark report saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all 90 videos, N = benchmark first N videos")
    parser.add_argument("--encoder", choices=["fast_spatial", "open_clip"], default="fast_spatial", help="Feature encoder: fast_spatial or open_clip")
    parser.add_argument("--device", default="cuda", help="Compute device: cuda or cpu (default: cuda)")
    parser.add_argument("--output", default="videomme_benchmark_results.json")
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.max_videos, args.output, encoder=args.encoder, device=args.device)