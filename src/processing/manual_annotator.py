"""Annotate a video using manual segments + known move list."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ManualSegments:
    video_path: Path
    fps: float
    segments: list[tuple[int, int]]  # (start_frame,end_frame) in sampled-frame indices

    @classmethod
    def from_json(cls, path: Path) -> "ManualSegments":
        obj = json.loads(path.read_text())
        segments = [(int(s["start_frame"]), int(s["end_frame"])) for s in obj["segments"]]
        return cls(video_path=Path(obj["video_path"]), fps=float(obj["fps"]), segments=segments)


def _get_font(size: int = 28) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _segment_index_at(frame_idx: int, segments: list[tuple[int, int]]) -> int | None:
    for i, (s, e) in enumerate(segments):
        if s <= frame_idx < e:
            return i
    return None


def annotate_with_manual_segments(
    manual_json: Path,
    moves: list[str],
    out_path: Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> Path:
    """Render annotated MP4 using manual segments."""
    ms = ManualSegments.from_json(manual_json)
    video_path = ms.video_path
    fps_used = ms.fps
    segments = ms.segments

    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = _get_font(28)

    # Prefer OpenCV if available (fast).
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        cv2 = None

    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required to render annotated MP4s on this setup.\n"
            "Install with:\n"
            "  python -m pip install opencv-python\n"
            "Or re-run later on a machine with OpenCV."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, int(round(float(src_fps) / float(fps_used))))
    fps_out = float(src_fps) / float(step)

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError(f"Failed to read frames from {video_path}")
    h0, w0 = frame0.shape[:2]

    if crop is not None:
        x0, y0, x1, y1 = crop
        w = max(1, min(w0, x1) - max(0, x0))
        h = max(1, min(h0, y1) - max(0, y0))
        out_size = (w, h)
    else:
        out_size = (w0, h0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, out_size)

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sampled_idx = 0
        idx = 0
        wrote = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue

            if crop is not None:
                x0, y0, x1, y1 = crop
                frame_bgr = frame_bgr[max(0, y0) : min(h0, y1), max(0, x0) : min(w0, x1)]

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)
            w, h = img.size

            seg_i = _segment_index_at(sampled_idx, segments)
            active = seg_i is not None

            # Border
            border = max(6, int(min(w, h) * 0.02))
            color = (34, 139, 34) if active else (60, 60, 60)  # green vs gray
            draw.rectangle([0, 0, w, border], fill=color)
            draw.rectangle([0, h - border, w, h], fill=color)
            draw.rectangle([0, 0, border, h], fill=color)
            draw.rectangle([w - border, 0, w, h], fill=color)

            # Text panel
            draw.rectangle([10, 10, min(w - 10, 520), 110], fill=(0, 0, 0))
            t_s = sampled_idx / float(fps_used)
            draw.text((22, 18), f"t={t_s:5.2f}s", fill=(255, 255, 255), font=font)

            if seg_i is not None:
                move = moves[seg_i] if seg_i < len(moves) else "?"
                draw.text(
                    (22, 54),
                    f"MOVE {seg_i+1:02d}: {move}  (ACTIVE)",
                    fill=(255, 255, 255),
                    font=font,
                )
            else:
                # show next move as cue
                next_i = 0
                for j, (s, _e) in enumerate(segments):
                    if sampled_idx < s:
                        next_i = j
                        break
                else:
                    next_i = len(segments)
                nxt = moves[next_i] if next_i < len(moves) else "-"
                draw.text((22, 54), f"REST  next: {next_i+1:02d} {nxt}", fill=(220, 220, 220), font=font)

            out_rgb = np.asarray(img, dtype=np.uint8)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)

            wrote += 1
            if wrote % 300 == 0:
                print(f"  wrote {wrote} frames...")

            sampled_idx += 1
            idx += 1

    finally:
        cap.release()
        writer.release()

    return out_path

