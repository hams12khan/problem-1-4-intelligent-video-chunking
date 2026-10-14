#!/usr/bin/env python3
"""Reliable streaming downloader for aintropy-ai/video-mme-sample10 dataset using requests."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import requests


BASE_REPO = "https://huggingface.co/datasets/aintropy-ai/video-mme-sample10"
API_TREE = "https://huggingface.co/api/datasets/aintropy-ai/video-mme-sample10/tree/main"


def download_file(url: str, dest_path: Path, desc: str = ""):
    """Streams a file download to dest_path with size check and resume."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    headers = {}
    downloaded_bytes = 0
    if temp_path.exists():
        downloaded_bytes = temp_path.stat().st_size
        headers["Range"] = f"bytes={downloaded_bytes}-"

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    
    # Handle server not supporting partial content
    if resp.status_code == 416 or (downloaded_bytes > 0 and resp.status_code == 200):
        # Start over
        downloaded_bytes = 0
        temp_path.unlink(missing_ok=True)
        resp = requests.get(url, stream=True, timeout=60)

    if resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {resp.status_code} when downloading {url}")

    total_bytes = int(resp.headers.get("content-length", 0)) + downloaded_bytes
    mode = "ab" if downloaded_bytes > 0 else "wb"

    t0 = time.time()
    last_print = t0
    cur_bytes = downloaded_bytes

    with open(temp_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=512 * 1024):
            if chunk:
                f.write(chunk)
                cur_bytes += len(chunk)
                now = time.time()
                if now - last_print > 1.0 or cur_bytes == total_bytes:
                    last_print = now
                    pct = (cur_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
                    mb = cur_bytes / (1024 * 1024)
                    tot_mb = total_bytes / (1024 * 1024)
                    speed = (cur_bytes - downloaded_bytes) / (now - t0 + 1e-4) / (1024 * 1024)
                    sys.stdout.write(f"\r  {desc}: {mb:.1f}/{tot_mb:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s   ")
                    sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    temp_path.replace(dest_path)


def download_dataset(data_dir: str, num_videos: int = 10, download_all: bool = False):
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"=== VideoMME-Sample10 Downloader ===")
    print(f"Target directory: {root}")

    # 1. Download metadata files
    print("\n[1/3] Downloading metadata and sample list...")
    for meta_file in ["sample_ids.txt", "manifest.json", "README.md"]:
        dest = root / meta_file
        if not dest.exists():
            url = f"{BASE_REPO}/raw/main/{meta_file}"
            print(f"  Downloading {meta_file}...")
            download_file(url, dest, meta_file)
        else:
            print(f"  {meta_file} already present.")

    # 2. Download all subtitles
    print("\n[2/3] Fetching subtitles index...")
    sub_api_url = f"{API_TREE}/subtitles"
    try:
        r = requests.get(sub_api_url, timeout=30)
        sub_items = [item["path"] for item in r.json() if item.get("path", "").endswith(".srt")]
        print(f"Found {len(sub_items)} subtitles. Syncing subtitles...")
        sub_dir = root / "subtitles"
        sub_dir.mkdir(parents=True, exist_ok=True)
        for s_path in sub_items:
            sub_name = os.path.basename(s_path)
            dest = sub_dir / sub_name
            if not dest.exists() or dest.stat().st_size == 0:
                s_url = f"{BASE_REPO}/raw/main/{s_path}"
                download_file(s_url, dest, sub_name)
        print("  All subtitles synced successfully.")
    except Exception as e:
        print(f"  Warning: Subtitle sync encountered: {e}")

    # 3. Read video IDs
    sample_ids_path = root / "sample_ids.txt"
    if not sample_ids_path.exists():
        print("[ERROR] sample_ids.txt missing!", file=sys.stderr)
        return

    with open(sample_ids_path, "r", encoding="utf-8") as f:
        all_ids = [line.strip() for line in f if line.strip()]

    target_ids = all_ids if download_all or num_videos <= 0 else all_ids[:num_videos]
    print(f"\n[3/3] Downloading {len(target_ids)} videos out of {len(all_ids)} available...")

    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for idx, vid in enumerate(target_ids, 1):
        dest = videos_dir / f"{vid}.mp4"
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"  [{idx}/{len(target_ids)}] {vid}.mp4 exists ({dest.stat().st_size / (1024*1024):.1f} MB). Skipping.")
            continue

        url = f"{BASE_REPO}/resolve/main/videos/{vid}.mp4"
        desc = f"[{idx}/{len(target_ids)}] {vid}.mp4"
        try:
            download_file(url, dest, desc)
        except Exception as e:
            print(f"  [ERROR] Failed to download {vid}.mp4: {e}")

    print(f"\n[COMPLETED] VideoMME-Sample10 download finished at: {root}")
    # Summary of downloaded files
    v_downloaded = list(videos_dir.glob("*.mp4"))
    s_downloaded = list((root / "subtitles").glob("*.srt"))
    print(f"Downloaded {len(v_downloaded)} videos and {len(s_downloaded)} subtitles ready for benchmarking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliable downloader for VideoMME-Sample10 using requests.")
    parser.add_argument("--data-dir", default="./data", help="Target destination directory (default: ./data)")
    parser.add_argument("--num-videos", type=int, default=10, help="Number of videos to download (default: 10)")
    parser.add_argument("--all", action="store_true", help="Download all 90 videos")
    args = parser.parse_args()
    download_dataset(args.data_dir, num_videos=args.num_videos, download_all=args.all)
