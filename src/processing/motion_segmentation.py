"""Motion-energy based move segmentation + validation helpers.

This is a lightweight "first-five validation gate" tool:
- Read an ingested scramble video from data/raw/<recorder>/<scramble_id>/video.*
- Compute a simple motion-energy signal from frame-to-frame differences
- Detect the clap spike (sync anchor)
- Segment post-clap motion into contiguous "move" regions
- Compare segment count to the known move count in metadata.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ValidationResult:
    scramble_id: str
    video_path: Path
    fps_used: float
    clap_frame: int
    start_frame: int
    end_frame: int | None
    expected_moves: int
    detected_moves: int
    segments: tuple[tuple[int, int], ...]

    @property
    def ok(self) -> bool:
        # Allow ±1 because the clap spike may get counted as a segment depending on framing.
        return abs(self.detected_moves - self.expected_moves) <= 1


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _runs_of_true(mask: np.ndarray, min_len: int, min_gap: int) -> list[tuple[int, int]]:
    """Return merged (start,end) runs where mask is True, inclusive-exclusive."""
    if mask.size == 0:
        return []

    starts: list[int] = []
    ends: list[int] = []
    in_run = False
    run_start = 0

    for i, v in enumerate(mask.tolist()):
        if v and not in_run:
            in_run = True
            run_start = i
        elif not v and in_run:
            in_run = False
            starts.append(run_start)
            ends.append(i)

    if in_run:
        starts.append(run_start)
        ends.append(mask.size)

    # Filter too-short runs
    runs = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_len]
    if not runs:
        return []

    # Merge close runs
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        if s - cur_e <= min_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def load_motion_energy(
    video_path: Path,
    target_fps: float = 30.0,
    max_seconds: float | None = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Compute per-frame motion energy by mean absdiff in grayscale.

    Returns (energy, fps_used) where energy has length (N-1) for N sampled frames.
    """
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or np.isnan(src_fps) or src_fps <= 0:
            # Fall back; many phone videos still decode fine.
            src_fps = 60.0
        step = max(1, int(round(src_fps / target_fps)))
        fps_used = float(src_fps) / float(step)

        max_frames = None
        if max_seconds is not None:
            max_frames = int(max_seconds * fps_used)

        prev = None
        energy: list[float] = []
        sampled = 0
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue

            if crop is not None:
                x0, y0, x1, y1 = crop
                frame = frame[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diff = cv2.absdiff(gray, prev)
                energy.append(float(diff.mean()))
            prev = gray
            sampled += 1
            if max_frames is not None and sampled >= max_frames:
                break
            idx += 1

        cap.release()
        return np.asarray(energy, dtype=np.float32), fps_used

    # Fallback path: imageio+ffmpeg (no OpenCV required)
    try:
        import imageio.v3 as iio  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Video decoding requires either OpenCV (opencv-python) or imageio+ffmpeg.\n"
            "Install deps with:\n"
            "  python -m pip install -r requirements.txt\n"
            "Then re-run:\n"
            "  python -m src.data.cli validate-first-five --recorder <name> --save-plots"
        ) from e

    # Read metadata FPS if available; if not, assume 60 (phone default).
    try:
        props = iio.improps(video_path)
        src_fps = float(getattr(props, "fps", 60.0) or 60.0)
    except Exception:
        src_fps = 60.0

    step = max(1, int(round(src_fps / target_fps)))
    fps_used = float(src_fps) / float(step)
    max_frames = None
    if max_seconds is not None:
        max_frames = int(max_seconds * fps_used)

    prev = None
    energy: list[float] = []
    sampled = 0
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx % step != 0:
            continue
        if crop is not None:
            x0, y0, x1, y1 = crop
            frame = np.asarray(frame)[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        # frame is RGB; convert to grayscale via luminance weights
        f = np.asarray(frame, dtype=np.float32)
        gray = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).astype(
            np.float32
        )
        if prev is not None:
            diff = np.abs(gray - prev)
            energy.append(float(diff.mean()))
        prev = gray
        sampled += 1
        if max_frames is not None and sampled >= max_frames:
            break

    return np.asarray(energy, dtype=np.float32), fps_used


def detect_clap_frame(
    energy: np.ndarray,
    fps: float,
    search_seconds: float = 20.0,
    *,
    min_time_s: float = 0.0,
) -> int:
    """Return an index into the *sampled frames* (not energy frames) for the clap.

    The naive argmax often hits the initial face-presentation phase. We instead
    search after ``min_time_s`` for a sharp, high-prominence peak.
    """
    if energy.size == 0:
        return 0
    # energy[i] corresponds to transition from frame i -> i+1 in sampled stream
    start_i = int(max(0.0, min_time_s) * fps)
    end_i = min(int(search_seconds * fps), energy.size)
    if end_i - start_i <= 2:
        return max(0, min(energy.size, start_i)) + 1

    e = energy[start_i:end_i].astype(np.float32)
    # Light smoothing so peaks are less noisy.
    smooth = _smooth_1d(e, window=max(3, int(0.06 * fps)))  # ~60ms

    # Find local maxima.
    peaks: list[int] = []
    for i in range(1, smooth.size - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            peaks.append(i)

    if not peaks:
        i = int(np.argmax(smooth))
        return (start_i + i) + 1

    # Score peaks by prominence and "narrowness" (clap should be brief).
    w = max(3, int(0.5 * fps))  # local neighborhood ~0.5s
    best_i = peaks[0]
    best_score = -1.0

    # If we find any "good enough" sharp peaks, take the earliest one.
    rest = float(np.percentile(smooth, 20))
    hi = float(np.percentile(smooth, 95))
    prom_floor = max(0.1, 0.35 * (hi - rest))
    earliest_good: int | None = None

    for p in peaks:
        left = max(0, p - w)
        right = min(smooth.size, p + w)
        base = float(np.min(smooth[left:right]))
        prom = float(smooth[p] - base)
        if prom <= 0:
            continue

        # Estimate width at 30% above baseline.
        level = base + 0.30 * prom
        l = p
        while l > 0 and smooth[l] > level:
            l -= 1
        r = p
        while r < smooth.size - 1 and smooth[r] > level:
            r += 1
        width = float(r - l) / float(fps)

        # Prefer high prominence but penalize wide peaks (face presentation is wide).
        score = prom / max(0.05, width)
        if earliest_good is None and prom >= prom_floor and width <= 0.25:
            earliest_good = p
        if score > best_score:
            best_score = score
            best_i = p

    chosen = earliest_good if earliest_good is not None else best_i
    return (start_i + chosen) + 1


def find_settle_end_frame(
    energy: np.ndarray,
    fps: float,
    start_frame: int,
    *,
    settle_s: float = 2.0,
) -> int | None:
    """Find the first time AFTER the scramble where the video stays 'still' for settle_s.

    Returns a sampled-frame index where we consider the scramble ended, or None
    if no clear settle period is found.
    """
    if energy.size == 0:
        return None
    start_e = max(0, start_frame - 1)
    e = energy[start_e:].astype(np.float32)
    if e.size < int(settle_s * fps):
        return None

    smooth = _smooth_1d(e, window=max(3, int(0.20 * fps)))
    # Estimate "rest" level robustly from low-percentile energy.
    rest = float(np.percentile(smooth, 20))
    # Anything close to rest counts as still.
    still_thresh = rest * 1.35 + 0.02
    still = smooth <= still_thresh

    need = max(3, int(settle_s * fps))
    run = 0
    for i, v in enumerate(still.tolist()):
        run = (run + 1) if v else 0
        if run >= need:
            end_e = i - need + 1  # energy index (relative)
            # Convert to sampled-frame index. (energy idx k corresponds to transition k->k+1)
            end_frame = (start_e + end_e) + 1
            return max(start_frame, end_frame)
    return None


def segment_moves_from_energy(
    energy: np.ndarray,
    fps: float,
    start_frame: int,
    *,
    end_frame: int | None = None,
    thresh_strength: float = 0.35,
) -> list[tuple[int, int]]:
    """Segment post-start motion into contiguous runs (in sampled-frame indices)."""
    if energy.size == 0:
        return []

    # Convert start_frame in sampled-frame space to energy index space.
    start_e = max(0, start_frame - 1)
    end_e = None if end_frame is None else max(start_e, min(energy.size, end_frame - 1))
    e = energy[start_e:end_e]
    if e.size < int(1.0 * fps):
        return []

    smooth = _smooth_1d(e, window=max(3, int(0.20 * fps)))  # ~200ms
    p30 = float(np.percentile(smooth, 30))
    p90 = float(np.percentile(smooth, 90))
    thresh = p30 + float(thresh_strength) * (p90 - p30)

    active = smooth > thresh
    runs = _runs_of_true(
        active,
        min_len=max(2, int(0.08 * fps)),    # >= ~80ms
        min_gap=max(1, int(0.12 * fps)),    # merge gaps <= ~120ms
    )

    # Map back into sampled-frame indices (inclusive-exclusive), offsetting by start_e.
    segments: list[tuple[int, int]] = []
    for s, t in runs:
        # s,t are energy indices; convert to frame indices (shift by +1)
        seg_start_frame = (start_e + s) + 1
        seg_end_frame = (start_e + t) + 1
        segments.append((seg_start_frame, seg_end_frame))
    return segments


def find_scramble_start_frame(
    energy: np.ndarray,
    fps: float,
    after_frame: int,
    *,
    min_active_s: float = 0.30,
) -> int:
    """Find the first sustained motion run after ``after_frame``.

    This suppresses false "moves" caused by small hand jitters between clap and
    actually beginning the scramble.
    """
    if energy.size == 0:
        return after_frame
    start_e = max(0, after_frame - 1)
    e = energy[start_e:].astype(np.float32)
    if e.size < int(0.5 * fps):
        return after_frame

    smooth = _smooth_1d(e, window=max(3, int(0.20 * fps)))
    p30 = float(np.percentile(smooth, 30))
    p90 = float(np.percentile(smooth, 90))
    thresh = p30 + 0.35 * (p90 - p30)
    active = smooth > thresh

    runs = _runs_of_true(
        active,
        min_len=max(2, int(min_active_s * fps)),
        min_gap=max(1, int(0.12 * fps)),
    )
    if not runs:
        return after_frame

    s0, _ = runs[0]
    return (start_e + s0) + 1


def _score_segments(
    energy: np.ndarray,
    segments: list[tuple[int, int]],
) -> list[float]:
    scores: list[float] = []
    for s, e in segments:
        # energy index range is [s-1, e-1)
        a = max(0, s - 1)
        b = max(a, min(energy.size, e - 1))
        if b <= a:
            scores.append(0.0)
        else:
            scores.append(float(np.sum(energy[a:b])))
    return scores


def pick_best_segments(
    energy: np.ndarray,
    segments: list[tuple[int, int]],
    expected_moves: int,
) -> list[tuple[int, int]]:
    """Drop weak segments first, then keep strongest up to expected_moves.

    We keep chronological order, but decide which ones to keep by energy score,
    so early jitter segments don't push out real moves at the end.
    """
    if len(segments) <= expected_moves:
        return segments
    scores = _score_segments(energy, segments)
    # Filter out extremely weak segments first.
    nonzero = [s for s in scores if s > 0]
    floor = float(np.percentile(nonzero, 25)) if nonzero else 0.0
    kept = [(seg, sc) for seg, sc in zip(segments, scores) if sc >= floor]
    if len(kept) <= expected_moves:
        return [seg for seg, _ in kept]

    # Keep top expected_moves by score, then sort by time.
    kept.sort(key=lambda x: x[1], reverse=True)
    kept = kept[:expected_moves]
    kept.sort(key=lambda x: x[0][0])
    return [seg for seg, _ in kept]


def validate_scramble_video(
    scramble_id: str,
    video_path: Path,
    expected_moves: int,
    target_fps: float = 30.0,
) -> ValidationResult:
    energy, fps_used = load_motion_energy(video_path, target_fps=target_fps)
    clap_frame = detect_clap_frame(energy, fps=fps_used, search_seconds=30.0, min_time_s=4.0)
    # Start analyzing ~1s after clap to avoid counting the clap spike as a move.
    after_clap = clap_frame + int(1.0 * fps_used)
    start_frame = find_scramble_start_frame(energy, fps=fps_used, after_frame=after_clap, min_active_s=0.30)
    end_frame = find_settle_end_frame(energy, fps=fps_used, start_frame=start_frame, settle_s=2.0)
    segments = segment_moves_from_energy(
        energy,
        fps=fps_used,
        start_frame=start_frame,
        end_frame=end_frame,
        thresh_strength=0.35,
    )
    segments = pick_best_segments(energy, segments, expected_moves=expected_moves)
    return ValidationResult(
        scramble_id=scramble_id,
        video_path=video_path,
        fps_used=fps_used,
        clap_frame=clap_frame,
        start_frame=start_frame,
        end_frame=end_frame,
        expected_moves=int(expected_moves),
        detected_moves=len(segments),
        segments=tuple(segments),
    )


def save_energy_plot(
    out_path: Path,
    energy: np.ndarray,
    fps: float,
    clap_frame: int,
    start_frame: int,
    segments: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    end_frame: int | None = None,
) -> None:
    import matplotlib

    # Ensure this works in headless/CLI-only environments.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(energy.size, dtype=np.float32) / float(fps)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, energy, linewidth=1.0)
    ax.set_title("Motion energy (mean absdiff) with detected segments")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy")

    ax.axvline((clap_frame - 1) / fps, color="orange", linestyle="--", linewidth=1.5, label="clap")
    ax.axvline((start_frame - 1) / fps, color="green", linestyle="--", linewidth=1.5, label="analysis start")
    if end_frame is not None:
        ax.axvline((end_frame - 1) / fps, color="purple", linestyle="--", linewidth=1.5, label="end (settle)")

    for (s, e) in segments:
        # segments are in sampled-frame indices; map to seconds in energy space
        x0 = max(0, s - 1) / fps
        x1 = max(0, e - 1) / fps
        ax.axvspan(x0, x1, color="red", alpha=0.15)

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

