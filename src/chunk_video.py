#!/usr/bin/env python3
"""Production CLI for Intelligent Video Chunking (Task 4 / Problem 1.4)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.chunker.config import ChunkerConfig
from src.chunker.pipeline import IntelligentVideoChunker


def find_matching_srt(video_path: str, explicit_srt: Optional[str] = None) -> str:
    """Locates a matching .srt file in the same directory or sibling 'subtitles/' directories."""
    if explicit_srt and os.path.exists(explicit_srt):
        return explicit_srt

    v_path = Path(video_path).resolve()
    stem = v_path.stem
    parent = v_path.parent

    # 1. Check same directory as the video file
    for ext in [".srt", ".en.srt", ".vtt"]:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)

    # 2. Check sibling subtitles directory (e.g., data/videos/x.mp4 -> data/subtitles/x.srt)
    for sub_dir_name in ["subtitles", "srt", "captions"]:
        sibling_sub = parent.parent / sub_dir_name
        if sibling_sub.is_dir():
            for ext in [".srt", ".en.srt", ".vtt"]:
                candidate = sibling_sub / f"{stem}{ext}"
                if candidate.exists():
                    return str(candidate)

    return ""


def process_path(
    input_arg: str,
    srt_arg: Optional[str],
    config: ChunkerConfig,
    output_arg: str,
):
    """Processes a single video file or a directory of videos with C-MASP."""
    chunker = IntelligentVideoChunker(config)
    if config.semantic_encoder == "open_clip":
        try:
            chunker.semantic_extractor._load_open_clip_if_needed()
        except Exception as e:
            print(
                f"[WARNING] Could not load open_clip ({e}); falling back to fast_spatial.",
                file=sys.stderr,
            )

    input_path = Path(input_arg)
    video_files = []
    if input_path.is_file():
        video_files.append(str(input_path))
    elif input_path.is_dir():
        for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"):
            video_files.extend([str(p) for p in sorted(input_path.rglob(ext))])
    else:
        print(f"[ERROR] Input path does not exist: {input_arg}", file=sys.stderr)
        sys.exit(1)

    if not video_files:
        print(f"[ERROR] No video files found in {input_arg}", file=sys.stderr)
        sys.exit(1)

    results = []
    for v_path in video_files:
        explicit_srt = srt_arg if len(video_files) == 1 else None
        srt_path = find_matching_srt(v_path, explicit_srt)
        res = chunker.chunk_video(v_path, srt_path=srt_path if srt_path else None)
        results.append(res.to_dict())
        print(
            f"[OK] {os.path.basename(v_path)} | Profile: {res.detected_profile} | "
            f"Duration: {res.video_duration_sec:.1f}s | Chunks: {len(res.chunks)} | "
            f"Speed: {res.realtime_factor_xrt:.1f}x realtime | "
            f"SCR: {res.metrics['SCR']:.3f} | DBFI: {res.metrics['DBFI']:.3f} | "
            f"Q_chunk: {res.metrics['Q_chunk']:.3f}"
        )

    payload = results[0] if len(results) == 1 else {"videos": results}
    out_dir = os.path.dirname(os.path.abspath(output_arg))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_arg, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved structured chunk telemetry to: {output_arg}")


def main():
    parser = argparse.ArgumentParser(
        description="C-MASP: Cascaded Multi-Modal Adaptive Semantic Video Chunker"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file (.mp4/.mkv/.avi) or directory of videos.",
    )
    parser.add_argument(
        "--srt",
        default=None,
        help="Optional path to .srt subtitle file (auto-detected if same stem).",
    )
    parser.add_argument(
        "--min-sec",
        type=float,
        default=10.0,
        help="Minimum allowed chunk duration in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=45.0,
        help="Maximum allowed chunk duration in seconds (default: 45.0).",
    )
    parser.add_argument(
        "--target-sec",
        type=float,
        default=20.0,
        help="Target optimal chunk duration in seconds (default: 20.0).",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Sparse frame sampling rate in FPS (default: 2.0).",
    )
    parser.add_argument(
        "--encoder",
        choices=["fast_spatial", "open_clip"],
        default="fast_spatial",
        help="Semantic feature extractor mode (default: fast_spatial).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run neural encoder on: cuda or cpu (default: cuda).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="chunks_output.json",
        help="Output JSON file path (default: chunks_output.json).",
    )

    args = parser.parse_args()

    config = ChunkerConfig(
        min_chunk_sec=args.min_sec,
        max_chunk_sec=args.max_sec,
        target_chunk_sec=args.target_sec,
        sample_fps=args.sample_fps,
        semantic_encoder=args.encoder,
        device=args.device,
    )
    process_path(args.input, args.srt, config, args.output)


if __name__ == "__main__":
    main()
