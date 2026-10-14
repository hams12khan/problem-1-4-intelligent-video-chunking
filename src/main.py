#!/usr/bin/env python3
"""Main Entrypoint for AIntropy Intelligent Video Chunking (C-MASP).

Supported execution modes:
1. Production Single Video:
   python src/main.py --input path/to/video.mp4 --output chunks.json
2. Directory Batch Processing:
   python src/main.py --input path/to/videos_folder/ --output batch_chunks.json
3. Evaluation Benchmark:
   python src/main.py --benchmark --data-dir path/to/dataset --output benchmark_report.json
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root and src/ to sys.path so imports work regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.chunker.config import ChunkerConfig
from src.chunker.pipeline import IntelligentVideoChunker


def main():
    parser = argparse.ArgumentParser(
        description="C-MASP: Cascaded Multi-Modal Adaptive Semantic Video Chunker (AIntropy Problem 1.4)"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to input video file (.mp4/.mkv/.avi) or directory of videos.",
    )
    parser.add_argument(
        "--srt",
        default=None,
        help="Optional path to .srt subtitle file (auto-detected if sibling/matching name).",
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
        "--encoder",
        choices=["fast_spatial", "open_clip"],
        default="fast_spatial",
        help="Semantic feature extractor: 'fast_spatial' (0 MB CPU) or 'open_clip' (605 MB GPU).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device: 'cpu' or 'cuda' (default: cpu).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="production_chunks.json",
        help="Path to output JSON file (default: production_chunks.json).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full 3-way ablation benchmark suite (Fixed-30s, Visual-Cut, C-MASP).",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing benchmark dataset videos and subtitles (default: ./data).",
    )

    args = parser.parse_args()

    # If run with --benchmark, invoke the benchmark suite
    if args.benchmark:
        try:
            from src.benchmark_videomme import run_benchmark
        except ImportError:
            from benchmark_videomme import run_benchmark
        run_benchmark(
            data_dir=args.data_dir,
            max_videos=0,
            output_json=args.output if args.output != "production_chunks.json" else "benchmark_report.json",
            encoder=args.encoder,
            device=args.device,
        )
        return

    # If no input is specified, print help and informative message
    if not args.input:
        parser.print_help()
        print("\n" + "=" * 80)
        print("C-MASP Intelligent Video Chunking Ready.")
        print("Example usage:")
        print("  python src/main.py --input path/to/video.mp4 --encoder fast_spatial --device cpu")
        print("  python src/main.py --benchmark --data-dir ./data")
        print("=" * 80)
        sys.exit(0)

    # Otherwise forward to chunk_video
    try:
        from src.chunk_video import process_path
    except ImportError:
        from chunk_video import process_path

    cfg = ChunkerConfig(
        min_chunk_sec=args.min_sec,
        max_chunk_sec=args.max_sec,
        target_chunk_sec=args.target_sec,
        semantic_encoder=args.encoder,
        device=args.device,
    )
    process_path(args.input, args.srt, cfg, args.output)


if __name__ == "__main__":
    main()
