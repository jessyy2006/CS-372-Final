"""Render an annotated video for debugging segmentation.

Produces a downsampled MP4 where frames inside detected segments are highlighted
and labeled with the segment index. This is meant for the "first-five" debug loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.processing.motion_segmentation import (
    detect_clap_frame,
    find_settle_end_frame,
    load_motion_energy,
    segment_moves_from_energy,
)


@dataclass(frozen=True)
class AnnotateConfig:
    target_fps: float = 30.0
    max_seconds: float | None = None
    start_after_clap_s: float = 1.0
    clap_min_time_s: float = 4.0
    settle_s: float = 2.0
    crop: tuple[int, int, int, int] | None = None
    border_px: int = 10


def _get_font(size: int = 24) -> ImageFont.ImageFont:
    try:
        # Works on most Windows installs
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _segment_index_at(sampled_frame_idx: int, segments: list[tuple[int, int]]) -> int | None:
    for k, (s, e) in enumerate(segments):
        if s <= sampled_frame_idx < e:
            return k
    return None


def _draw_border(draw: ImageDraw.ImageDraw, w: int, h: int, px: int, color: tuple[int, int, int]) -> None:
    # top
    draw.rectangle([0, 0, w, px], fill=color)
    # bottom
    draw.rectangle([0, h - px, w, h], fill=color)
    # left
    draw.rectangle([0, 0, px, h], fill=color)
    # right
    draw.rectangle([w - px, 0, w, h], fill=color)


def _iter_sampled_frames_imageio(video_path: Path, src_fps: float, target_fps: float, max_seconds: float | None):
    import imageio.v3 as iio  # type: ignore

    step = max(1, int(round(src_fps / target_fps)))
    fps_used = float(src_fps) / float(step)
    max_frames = None if max_seconds is None else int(max_seconds * fps_used)

    sampled_idx = 0
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx % step != 0:
            continue
        arr = np.asarray(frame, dtype=np.uint8)
        # Ensure RGB HxWx3
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        yield sampled_idx, arr, fps_used
        sampled_idx += 1
        if max_frames is not None and sampled_idx >= max_frames:
            break


def _get_src_fps_imageio(video_path: Path) -> float:
    import imageio.v3 as iio  # type: ignore

    try:
        props = iio.improps(video_path)
        fps = float(getattr(props, "fps", 0.0) or 0.0)
        return fps if fps > 0 else 60.0
    except Exception:
        return 60.0


def annotate_segmentation_video(
    video_path: Path,
    out_path: Path,
    *,
    expected_moves: int | None = None,
    forced_clap_time_s: float | None = None,
    cfg: AnnotateConfig = AnnotateConfig(),
) -> dict[str, object]:
    """Write annotated MP4 to out_path. Returns summary info dict."""
    energy, fps_used = load_motion_energy(
        video_path,
        target_fps=cfg.target_fps,
        max_seconds=cfg.max_seconds,
        crop=cfg.crop,
    )
    clap_frame = detect_clap_frame(
        energy,
        fps=fps_used,
        search_seconds=30.0,
        min_time_s=cfg.clap_min_time_s,
    )
    if forced_clap_time_s is not None and forced_clap_time_s > 0:
        clap_frame = int(forced_clap_time_s * fps_used)
    start_frame = clap_frame + int(cfg.start_after_clap_s * fps_used)
    end_frame = find_settle_end_frame(energy, fps=fps_used, start_frame=start_frame, settle_s=cfg.settle_s)
    segments = segment_moves_from_energy(energy, fps=fps_used, start_frame=start_frame)
    if end_frame is not None:
        segments = [(s, e) for (s, e) in segments if s < end_frame]
    if expected_moves is not None and len(segments) > expected_moves:
        segments = segments[:expected_moves]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = _get_font(24)

    # Prefer OpenCV if available (much faster than imageio on some MOVs).
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        step = max(1, int(round(float(src_fps) / cfg.target_fps)))
        fps_out = float(src_fps) / float(step)

        # Read first sampled frame to get dimensions
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frames from {video_path}")
        h, w = frame_bgr.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (w, h))

        try:
            sampled_idx = 0
            idx = 0
            wrote = 0
            # We already consumed one frame at idx=0.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % step != 0:
                    idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(img)
                ww, hh = img.size

                seg_k = _segment_index_at(sampled_idx, segments)
                if seg_k is not None and sampled_idx >= start_frame:
                    _draw_border(draw, ww, hh, cfg.border_px, (220, 20, 60))
                    label = f"SEG {seg_k:02d}"
                elif sampled_idx < start_frame:
                    _draw_border(draw, ww, hh, cfg.border_px, (255, 165, 0))
                    label = "PRE (state/clap)"
                elif end_frame is not None and sampled_idx >= end_frame:
                    _draw_border(draw, ww, hh, cfg.border_px, (138, 43, 226))
                    label = "POST (ignore)"
                else:
                    label = "REST"

                t_s = sampled_idx / float(fps_out)
                draw.rectangle([10, 10, 320, 74], fill=(0, 0, 0))
                draw.text((20, 18), f"t={t_s:5.2f}s", fill=(255, 255, 255), font=font)
                draw.text((20, 44), label, fill=(255, 255, 255), font=font)

                out_rgb = np.asarray(img, dtype=np.uint8)
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)

                wrote += 1
                if wrote % 60 == 0:
                    print(f"  wrote {wrote} frames...")

                sampled_idx += 1
                if cfg.max_seconds is not None and (sampled_idx / fps_out) >= cfg.max_seconds:
                    break
                idx += 1
        finally:
            cap.release()
            writer.release()
    else:
        # Fallback: imageio writer
        import imageio  # type: ignore

        src_fps = _get_src_fps_imageio(video_path)
        writer = imageio.get_writer(str(out_path), fps=fps_used, codec="libx264", quality=7, macro_block_size=1)
        try:
            wrote = 0
            for sampled_idx, frame_rgb, _ in _iter_sampled_frames_imageio(
                video_path, src_fps=src_fps, target_fps=cfg.target_fps, max_seconds=cfg.max_seconds
            ):
                img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(img)
                w, h = img.size

                seg_k = _segment_index_at(sampled_idx, segments)
                if seg_k is not None and sampled_idx >= start_frame:
                    _draw_border(draw, w, h, cfg.border_px, (220, 20, 60))  # crimson
                    label = f"SEG {seg_k:02d}"
                elif sampled_idx < start_frame:
                    _draw_border(draw, w, h, cfg.border_px, (255, 165, 0))  # orange: pre-analysis
                    label = "PRE (state/clap)"
                elif end_frame is not None and sampled_idx >= end_frame:
                    _draw_border(draw, w, h, cfg.border_px, (138, 43, 226))  # purple: post-scramble
                    label = "POST (ignore)"
                else:
                    label = "REST"

                t_s = sampled_idx / float(fps_used)
                draw.rectangle([10, 10, 260, 70], fill=(0, 0, 0))
                draw.text((20, 18), f"t={t_s:5.2f}s", fill=(255, 255, 255), font=font)
                draw.text((20, 42), label, fill=(255, 255, 255), font=font)

                writer.append_data(np.asarray(img))
                wrote += 1
                if wrote % 60 == 0:
                    print(f"  wrote {wrote} frames...")
        finally:
            writer.close()

    return {
        "video": str(video_path),
        "out": str(out_path),
        "fps_used": fps_used,
        "clap_frame": clap_frame,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "detected_moves": len(segments),
    }

