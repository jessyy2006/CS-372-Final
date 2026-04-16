"""Manual move segmentation UI (spacebar hold) for debugging/labeling.

Controls:
- Space (hold): mark a move segment while held
- u: undo last segment
- s: save segments JSON
- q / Esc: quit

This is intentionally lightweight and uses Tkinter (bundled with Python).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class ManualSegmentation:
    video_path: str
    fps: float
    segments: list[Segment]

    def to_json(self) -> str:
        obj = {
            "video_path": self.video_path,
            "fps": self.fps,
            "segments": [asdict(s) for s in self.segments],
        }
        return json.dumps(obj, indent=2)


def _get_fps_imageio(video_path: Path) -> float:
    import imageio.v3 as iio  # type: ignore

    try:
        props = iio.improps(video_path)
        fps = float(getattr(props, "fps", 0.0) or 0.0)
        return fps if fps > 0 else 60.0
    except Exception:
        return 60.0


def _iter_frames(video_path: Path, *, step: int, max_frames: int | None):
    import imageio.v3 as iio  # type: ignore

    yielded = 0
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx % step != 0:
            continue
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        yield yielded, arr
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break


def run_manual_segmenter(
    video_path: Path,
    out_json: Path,
    *,
    target_fps: float = 30.0,
    skip_seconds: float = 0.0,
    speed: float = 1.0,
    max_seconds: float | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> Path:
    import tkinter as tk

    src_fps = _get_fps_imageio(video_path)
    step = max(1, int(round(src_fps / target_fps)))
    fps_used = float(src_fps) / float(step)
    max_frames = None if max_seconds is None else int(max_seconds * fps_used)
    skip_frames = int(max(0.0, skip_seconds) * fps_used)
    speed = float(speed) if speed and speed > 0 else 1.0

    segments: list[Segment] = []
    holding = {"down": False, "start": None}

    root = tk.Tk()
    root.title("Manual move segmentation (hold Space)")

    label = tk.Label(root)
    label.pack(expand=True, fill="both")

    status = tk.StringVar()
    status_label = tk.Label(root, textvariable=status, anchor="w", justify="left")
    status_label.pack(fill="x")

    def update_status(frame_idx: int):
        t_s = frame_idx / fps_used
        cur = ""
        if holding["down"] and holding["start"] is not None:
            cur = f"  HOLDING: start={holding['start']} (t={holding['start']/fps_used:.2f}s)"
        status.set(
            f"frame={frame_idx}  t={t_s:.2f}s  fps_used~{fps_used:.1f}  segments={len(segments)}{cur}\n"
            f"Controls: hold Space=segment | u=undo | s=save | q/Esc=quit"
        )

    def on_space_down(_evt=None):
        if holding["down"]:
            return
        holding["down"] = True
        holding["start"] = current["frame"]

    def on_space_up(_evt=None):
        if not holding["down"]:
            return
        holding["down"] = False
        start = holding["start"]
        end = current["frame"]
        holding["start"] = None
        if start is None:
            return
        if end <= start:
            return
        segments.append(Segment(start_frame=int(start), end_frame=int(end)))

    def on_undo(_evt=None):
        if segments:
            segments.pop()

    def on_save(_evt=None):
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            ManualSegmentation(video_path=str(video_path), fps=fps_used, segments=segments).to_json()
        )
        status.set(status.get() + f"\nSaved: {out_json}")

    def on_quit(_evt=None):
        root.destroy()

    root.bind("<KeyPress-space>", on_space_down)
    root.bind("<KeyRelease-space>", on_space_up)
    root.bind("u", on_undo)
    root.bind("s", on_save)
    root.bind("q", on_quit)
    root.bind("<Escape>", on_quit)

    # Simple playback loop driven by Tk "after" callbacks.
    current = {"frame": 0, "iter": _iter_frames(video_path, step=step, max_frames=max_frames), "boot": True}

    def tick():
        # Skip initial seconds by consuming frames without rendering.
        if current.get("boot"):
            current["boot"] = False
            try:
                for _ in range(skip_frames):
                    frame_idx, _ = next(current["iter"])
                    current["frame"] = frame_idx
            except StopIteration:
                on_save()
                on_quit()
                return

        try:
            frame_idx, rgb = next(current["iter"])
        except StopIteration:
            on_save()
            on_quit()
            return

        current["frame"] = frame_idx

        if crop is not None:
            x0, y0, x1, y1 = crop
            rgb = rgb[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        # On-video indicator while holding Space.
        if holding["down"] and holding["start"] is not None:
            w, h = img.size
            border = max(6, int(min(w, h) * 0.02))
            draw.rectangle([0, 0, w, border], fill=(220, 20, 60))
            draw.rectangle([0, h - border, w, h], fill=(220, 20, 60))
            draw.rectangle([0, 0, border, h], fill=(220, 20, 60))
            draw.rectangle([w - border, 0, w, h], fill=(220, 20, 60))
            draw.rectangle([10, 10, 260, 54], fill=(0, 0, 0))
            draw.text((20, 18), "RECORDING...", fill=(255, 255, 255))

        # Resize to fit screen (no cropping), especially for vertical phone videos.
        # Using conservative defaults so the whole frame is visible on typical laptops.
        max_w = 960
        max_h = 900
        w0, h0 = img.size
        scale = min(max_w / float(w0), max_h / float(h0), 1.0)
        if scale < 1.0:
            img = img.resize((int(w0 * scale), int(h0 * scale)))

        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo  # keep alive

        update_status(frame_idx)

        # Playback timing: speed=1.0 is "normal"; >1 is faster.
        delay_ms = max(1, int(1000 / (fps_used * speed)))
        root.after(delay_ms, tick)

    update_status(0)
    root.after(0, tick)
    root.mainloop()

    # If user quit without saving, still save whatever we have.
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        ManualSegmentation(video_path=str(video_path), fps=fps_used, segments=segments).to_json()
    )
    return out_json

