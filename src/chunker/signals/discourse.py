"""Tier 2: Acoustic & Linguistic Discourse Alignment (De-Overlapped SRT, VAD & Snapping)."""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class UtteranceInterval:
    """Represents a spoken discourse segment from subtitles or VAD."""

    start_sec: float
    end_sec: float
    text: str = ""
    ends_sentence: bool = True


class DiscourseSignalExtractor:
    """Parses subtitle files (.srt), resolves rolling caption overlaps, and snaps boundaries."""

    TIMESTAMP_PATTERN = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    NON_SPEECH_TAG_PATTERN = re.compile(
        r"\[(?:music|applause|laughter|silence|noise|cheering|♪|sound)[^\]]*\]"
        r"|\((?:music|applause|laughter|silence|noise|cheering)[^\)]*\)|♪+",
        flags=re.IGNORECASE,
    )

    def __init__(self, boundary_margin_sec: float = 0.25):
        self.boundary_margin_sec = boundary_margin_sec

    @staticmethod
    def _hms_to_sec(h: str, m: str, s: str, ms: str) -> float:
        return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0

    @staticmethod
    def _deduplicate_rolling_text(prev_text: str, curr_text: str) -> str:
        """Removes duplicate rolling caption lines repeated between consecutive YouTube cues."""
        if not prev_text or not curr_text:
            return curr_text
        prev_words = prev_text.split()
        curr_words = curr_text.split()
        max_overlap = min(len(prev_words), len(curr_words), 18)
        for k in range(max_overlap, 1, -1):
            if [w.lower() for w in prev_words[-k:]] == [w.lower() for w in curr_words[:k]]:
                remainder = " ".join(curr_words[k:]).strip()
                return remainder if remainder else curr_text
        return curr_text

    def parse_srt(self, srt_path: Optional[str]) -> List[UtteranceInterval]:
        """Parses a SubRip (.srt) file and resolves YouTube 2-line rolling caption overlaps."""
        if not srt_path or not os.path.exists(srt_path):
            return []

        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        blocks = re.split(r"\n\s*\n", content.strip())
        raw_intervals: List[UtteranceInterval] = []

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue

            match = None
            ts_line_idx = -1
            for idx, line in enumerate(lines):
                m = self.TIMESTAMP_PATTERN.search(line)
                if m:
                    match = m
                    ts_line_idx = idx
                    break

            if not match:
                continue

            start_s = self._hms_to_sec(*match.groups()[:4])
            end_s = self._hms_to_sec(*match.groups()[4:])
            if end_s <= start_s:
                continue

            raw_text = " ".join(lines[ts_line_idx + 1 :])
            clean_text = re.sub(r"<[^>]+>", "", raw_text)
            spoken_text = self.NON_SPEECH_TAG_PATTERN.sub("", clean_text)
            spoken_text = re.sub(r"\s+", " ", spoken_text).strip()

            # Skip pure background music / applause cues so they act as valid discourse pauses
            if not spoken_text:
                continue

            ends_sentence = bool(re.search(r"[.?!।\"'”’)]$", spoken_text))

            raw_intervals.append(
                UtteranceInterval(
                    start_sec=start_s,
                    end_sec=end_s,
                    text=spoken_text,
                    ends_sentence=ends_sentence,
                )
            )

        if not raw_intervals:
            return []

        # Sort by start timestamp and resolve rolling YouTube caption overlaps
        raw_intervals.sort(key=lambda u: (u.start_sec, u.end_sec))
        resolved: List[UtteranceInterval] = []

        for u in raw_intervals:
            if not resolved:
                resolved.append(u)
                continue

            prev = resolved[-1]
            dedup_text = self._deduplicate_rolling_text(prev.text, u.text)
            u.text = dedup_text

            if abs(u.start_sec - prev.start_sec) < 0.08:
                prev.end_sec = max(prev.end_sec, u.end_sec)
                if dedup_text and dedup_text not in prev.text:
                    prev.text = (prev.text + " " + dedup_text).strip()
                prev.ends_sentence = u.ends_sentence
                continue

            # Clamp previous cue's end_sec so rolling 2-line captions do not overlap into the next cue
            if prev.end_sec > u.start_sec:
                prev.end_sec = u.start_sec

            if u.end_sec - u.start_sec >= 0.12:
                resolved.append(u)

        return [u for u in resolved if (u.end_sec - u.start_sec) >= 0.12]

    def compute_audio_rms_vad(
        self,
        audio_samples: np.ndarray,
        sample_rate: int,
        frame_sec: float = 0.05,
        hop_sec: float = 0.025,
        energy_threshold_ratio: float = 0.15,
        min_speech_duration_sec: float = 0.25,
        min_silence_duration_sec: float = 0.35,
    ) -> List[UtteranceInterval]:
        """Zero-training Voice Activity Detector (VAD) via Short-Time Audio RMS Energy Envelope."""
        if len(audio_samples) == 0 or sample_rate <= 0:
            return []

        samples = audio_samples.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / (np.max(np.abs(samples)) + 1e-8)

        frame_len = max(1, int(round(frame_sec * sample_rate)))
        hop_len = max(1, int(round(hop_sec * sample_rate)))

        num_frames = max(1, (len(samples) - frame_len) // hop_len + 1)
        rms = np.zeros(num_frames, dtype=np.float32)

        for idx in range(num_frames):
            start = idx * hop_len
            end = start + frame_len
            chunk = samples[start:end]
            rms[idx] = float(np.sqrt(np.mean(chunk**2)))

        noise_floor = float(np.percentile(rms, 15))
        peak_energy = float(np.percentile(rms, 95))
        threshold = noise_floor + energy_threshold_ratio * max(
            peak_energy - noise_floor, 0.01
        )

        is_speech = rms >= threshold
        frame_times = np.arange(num_frames) * hop_sec

        raw_intervals: List[Tuple[float, float]] = []
        in_speech = False
        start_t = 0.0

        for idx, active in enumerate(is_speech):
            t = float(frame_times[idx])
            if active and not in_speech:
                in_speech = True
                start_t = t
            elif not active and in_speech:
                in_speech = False
                raw_intervals.append((start_t, t))

        if in_speech:
            raw_intervals.append((start_t, float(frame_times[-1] + hop_sec)))

        if not raw_intervals:
            return []

        merged: List[Tuple[float, float]] = []
        cur_s, cur_e = raw_intervals[0]

        for nxt_s, nxt_e in raw_intervals[1:]:
            if (nxt_s - cur_e) < min_silence_duration_sec:
                cur_e = nxt_e
            else:
                if (cur_e - cur_s) >= min_speech_duration_sec:
                    merged.append((cur_s, cur_e))
                cur_s, cur_e = nxt_s, nxt_e

        if (cur_e - cur_s) >= min_speech_duration_sec:
            merged.append((cur_s, cur_e))

        return [
            UtteranceInterval(
                start_sec=round(s, 2),
                end_sec=round(e, 2),
                text="[Audio RMS VAD Speech]",
                ends_sentence=True,
            )
            for s, e in merged
        ]

    def extract_utterances_from_audio(self, video_path: str) -> List[UtteranceInterval]:
        """Attempts to load audio accompanying the video and run RMS VAD."""
        base, _ = os.path.splitext(video_path)
        for ext in [".wav", ".audio.wav"]:
            wav_candidate = base + ext
            if os.path.exists(wav_candidate):
                try:
                    import scipy.io.wavfile as wavfile

                    sr, data = wavfile.read(wav_candidate)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    return self.compute_audio_rms_vad(data, sr)
                except Exception:
                    pass

        import shutil
        import subprocess

        if shutil.which("ffmpeg"):
            try:
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "s16le",
                    "-",
                ]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                raw_bytes, _ = proc.communicate(timeout=10)
                if proc.returncode == 0 and len(raw_bytes) > 0:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16)
                    return self.compute_audio_rms_vad(samples, sample_rate=16000)
            except Exception:
                pass

        return []

    def is_timestamp_fractured(
        self, t: float, utterances: List[UtteranceInterval], margin: Optional[float] = None
    ) -> bool:
        """Returns True if timestamp t strictly fractures ANY utterance interval."""
        if not utterances:
            return False
        m = self.boundary_margin_sec if margin is None else margin
        return any((u.start_sec + m) < t < (u.end_sec - m) for u in utterances)

    def compute_speech_penalty_grid(
        self, grid_timestamps: np.ndarray, utterances: List[UtteranceInterval]
    ) -> np.ndarray:
        """Evaluates the speech interruption penalty P_speech(t) in [0, 1] at every grid timestamp."""
        penalties = np.zeros_like(grid_timestamps, dtype=np.float32)
        if not utterances:
            return penalties

        margin = self.boundary_margin_sec
        for i, t_val in enumerate(grid_timestamps):
            t = float(t_val)
            if self.is_timestamp_fractured(t, utterances, margin=margin):
                dists = [
                    min(abs(t - u.start_sec), abs(t - u.end_sec))
                    for u in utterances
                    if (u.start_sec + margin) < t < (u.end_sec - margin)
                ]
                min_dist_to_edge = min(dists) if dists else 1.0
                if min_dist_to_edge <= 0.65:
                    penalties[i] = 0.45
                else:
                    penalties[i] = 1.0
            else:
                near_incomplete = any(
                    abs(t - u.end_sec) <= margin and not u.ends_sentence
                    for u in utterances
                )
                penalties[i] = 0.12 if near_incomplete else 0.0

        return penalties

    def snap_boundary_to_discourse_pause(
        self,
        candidate_sec: float,
        prev_start_sec: float,
        next_end_sec: float,
        utterances: List[UtteranceInterval],
        min_chunk_sec: float,
        max_chunk_sec: float,
        max_snap_sec: float = 3.0,
    ) -> float:
        """Snaps a coarse grid boundary timestamp to the nearest millisecond-exact speech pause.

        Prioritizes strictly exterior pause timestamps (margin=0.0, i.e., true silence gaps
        or exact utterance boundaries [u.end_sec, nxt_u.start_sec]) before falling back to
        the inner tolerance margin.
        """
        if not utterances:
            return candidate_sec

        # If already strictly outside all utterances (in a true silence gap), keep as-is
        if not self.is_timestamp_fractured(candidate_sec, utterances, margin=0.0):
            return candidate_sec

        lo_bound = max(
            prev_start_sec + min_chunk_sec,
            next_end_sec - max_chunk_sec,
            candidate_sec - max_snap_sec,
        )
        hi_bound = min(
            prev_start_sec + max_chunk_sec,
            next_end_sec - min_chunk_sec,
            candidate_sec + max_snap_sec,
        )

        if lo_bound > hi_bound:
            return candidate_sec

        # Build candidate set from exact utterance edges, silence gap midpoints, and 25ms grid
        candidates = []
        for idx, u in enumerate(utterances):
            for pt in (u.start_sec, u.end_sec):
                if lo_bound <= pt <= hi_bound:
                    candidates.append(pt)
            if idx + 1 < len(utterances):
                nxt_u = utterances[idx + 1]
                if nxt_u.start_sec >= u.end_sec:
                    mid = 0.5 * (u.end_sec + nxt_u.start_sec)
                    if lo_bound <= mid <= hi_bound:
                        candidates.append(mid)

        fine_grid = np.arange(lo_bound, hi_bound + 1e-4, 0.025)
        candidates.extend([float(x) for x in fine_grid])

        # Pass 1: Strictly exterior pause points (margin = 0.0 -> never inside an utterance)
        best_t = None
        best_dist = float("inf")
        for pt in candidates:
            if not self.is_timestamp_fractured(pt, utterances, margin=0.0):
                dist = abs(pt - candidate_sec)
                if dist < best_dist:
                    best_dist = dist
                    best_t = pt

        if best_t is not None:
            return round(float(best_t), 3)

        # Pass 2: Fallback to boundary_margin_sec if utterances completely span [lo_bound, hi_bound]
        margin = self.boundary_margin_sec
        best_t = candidate_sec
        best_dist = float("inf")
        for pt in candidates:
            if not self.is_timestamp_fractured(pt, utterances, margin=margin):
                dist = abs(pt - candidate_sec)
                if dist < best_dist:
                    best_dist = dist
                    best_t = pt

        return round(float(best_t), 3)

    @staticmethod
    def extract_chunk_transcript(
        start_sec: float, end_sec: float, utterances: List[UtteranceInterval]
    ) -> str:
        """Aggregates de-overlapped subtitle text overlapping with [start_sec, end_sec]."""
        if not utterances:
            return ""
        snippets = [
            u.text
            for u in utterances
            if u.end_sec > (start_sec + 0.15) and u.start_sec < (end_sec - 0.15)
        ]
        joined = " ".join(snippets)
        return joined[:180] + ("..." if len(joined) > 180 else "")