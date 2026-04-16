"""Command-line entry points for Day 1 data work.

Run from the repo root::

    python -m src.data.cli generate --count 200
    python -m src.data.cli list-pending
    python -m src.data.cli show 0001
    python -m src.data.cli ingest --scramble 0001 --recorder alice ~/Movies/IMG_1234.MOV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

from src.data.recording_utils import (
    ingest_recording,
    list_recorded,
    load_scramble,
    next_pending,
)
from src.data.scramble_generator import generate_batch
from src.processing.motion_segmentation import (
    load_motion_energy,
    save_energy_plot,
    validate_scramble_video,
)
from src.processing.video_annotator import AnnotateConfig, annotate_segmentation_video
from src.processing.manual_segmenter import run_manual_segmenter
from src.processing.manual_annotator import annotate_with_manual_segments

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAMBLES_DIR = REPO_ROOT / "data" / "scrambles"
RAW_ROOT = REPO_ROOT / "data" / "raw"

def _get_video_size(video_path: Path) -> tuple[int, int]:
    """Return (width,height) for the first decodable frame."""
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                return int(w), int(h)

    # Fallback: imageio
    import imageio.v3 as iio  # type: ignore

    for frame in iio.imiter(video_path):
        arr = frame
        try:
            h, w = arr.shape[:2]
            return int(w), int(h)
        except Exception:
            continue
    raise RuntimeError(f"Could not read any frames from {video_path}")


def _auto_crop_top_region(video_path: Path) -> tuple[int, int, int, int]:
    """Ignore top/left/right 5%; keep top ~60% height (with wiggle room)."""
    w, h = _get_video_size(video_path)
    x0 = int(0.05 * w)
    x1 = int(0.95 * w)
    y0 = int(0.05 * h)          # ignore top 5%
    y1 = int(0.60 * h)          # top 60% (extra wiggle below midline)
    return (x0, y0, x1, y1)


def _resolve_scramble_id(raw: str) -> str:
    """Accept either '0001' or 'scramble_0001' and normalize."""
    if raw.startswith("scramble_"):
        return raw
    return f"scramble_{int(raw):04d}"


def cmd_generate(args: argparse.Namespace) -> int:
    scrambles = generate_batch(
        count=args.count,
        output_dir=SCRAMBLES_DIR,
        n_moves=args.n_moves,
        master_seed=args.seed,
    )
    print(f"Wrote {len(scrambles)} scrambles to {SCRAMBLES_DIR}")
    print(f"Example (first): {scrambles[0].scramble_id}: {' '.join(scrambles[0].moves)}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    sid = _resolve_scramble_id(args.scramble_id)
    s = load_scramble(SCRAMBLES_DIR, sid)
    print(f"{s.scramble_id}  (seed={s.seed}, n_moves={s.n_moves})")
    print(" ".join(s.moves))
    return 0


def cmd_list_pending(_: argparse.Namespace) -> int:
    pending = next_pending(SCRAMBLES_DIR, RAW_ROOT)
    recorded = list_recorded(RAW_ROOT)
    total = len(pending) + len(recorded)
    print(f"Recorded: {len(recorded)} / {total}")
    print(f"Pending : {len(pending)}")
    if pending:
        print("\nNext 10 pending:")
        for sid in pending[:10]:
            s = load_scramble(SCRAMBLES_DIR, sid)
            print(f"  {s.scramble_id}: {' '.join(s.moves)}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    sid = _resolve_scramble_id(args.scramble)
    scramble = load_scramble(SCRAMBLES_DIR, sid)
    source = Path(args.source).expanduser().resolve()
    target_dir = ingest_recording(
        source_video=source,
        scramble=scramble,
        recorder=args.recorder,
        raw_root=RAW_ROOT,
        notes=args.notes or "",
        move=args.move,
    )
    print(f"Ingested {source.name} -> {target_dir}")
    return 0


def cmd_validate_first_five(args: argparse.Namespace) -> int:
    # Default to 0001-0005 if not provided.
    ids = args.scramble_ids or ["0001", "0002", "0003", "0004", "0005"]
    sids = [_resolve_scramble_id(x) for x in ids]

    validation_root = REPO_ROOT / "data" / "validation"
    any_fail = False

    for sid in sids:
        meta_path = RAW_ROOT / args.recorder / sid / "metadata.json"
        if not meta_path.exists():
            print(f"[SKIP] {sid}: no metadata at {meta_path}")
            any_fail = True
            continue
        meta = json.loads(meta_path.read_text())
        expected = int(meta["n_moves"])
        video_path = RAW_ROOT / args.recorder / sid / meta["video_filename"]
        if not video_path.exists():
            print(f"[SKIP] {sid}: no video at {video_path}")
            any_fail = True
            continue

        print(f"[RUN ] {sid}: computing motion energy + segments...")
        res = validate_scramble_video(
            scramble_id=sid,
            video_path=video_path,
            expected_moves=expected,
            target_fps=args.fps,
        )

        status = "PASS" if res.ok else "FAIL"
        print(
            f"[{status}] {sid}: expected {res.expected_moves}, detected {res.detected_moves} "
            f"(fps~{res.fps_used:.1f}, clap_frame={res.clap_frame}, start_frame={res.start_frame})"
        )

        if args.save_plots:
            energy, fps_used = load_motion_energy(video_path, target_fps=args.fps)
            out_path = validation_root / args.recorder / f"{sid}_energy.png"
            save_energy_plot(
                out_path=out_path,
                energy=energy,
                fps=fps_used,
                clap_frame=res.clap_frame,
                start_frame=res.start_frame,
                segments=list(res.segments),
                end_frame=res.end_frame,
            )
            print(f"  plot: {out_path}")

        any_fail = any_fail or (not res.ok)

    return 1 if any_fail else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.data.cli", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate a batch of scrambles")
    p_gen.add_argument("--count", type=int, default=200)
    p_gen.add_argument("--n-moves", type=int, default=25)
    p_gen.add_argument("--seed", type=int, default=0, help="Master seed for reproducibility")
    p_gen.set_defaults(func=cmd_generate)

    p_show = sub.add_parser("show", help="Print one scramble's moves")
    p_show.add_argument("scramble_id")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list-pending", help="Show scrambles not yet recorded")
    p_list.set_defaults(func=cmd_list_pending)

    p_ing = sub.add_parser("ingest", help="Move a phone recording into data/raw/")
    p_ing.add_argument("source", help="Path to the phone recording (.mov/.mp4)")
    p_ing.add_argument("--scramble", required=True, help="Scramble id (e.g. 0001)")
    p_ing.add_argument("--recorder", required=True, help="Recorder name (e.g. alice)")
    p_ing.add_argument("--notes", default="", help="Optional free-text notes")
    p_ing.add_argument("--move", action="store_true", help="Move instead of copy")
    p_ing.set_defaults(func=cmd_ingest)

    p_val = sub.add_parser(
        "validate-first-five",
        help="Run the first-five validation gate on ingested scrambles",
    )
    p_val.add_argument("--recorder", required=True, help="Recorder name (e.g. soham)")
    p_val.add_argument(
        "scramble_ids",
        nargs="*",
        help="Optional ids like 0001 0002 ... (defaults to 0001-0005)",
    )
    p_val.add_argument("--fps", type=float, default=30.0, help="Target FPS for analysis")
    p_val.add_argument(
        "--save-plots",
        action="store_true",
        help="Write motion-energy plots under data/validation/",
    )
    p_val.set_defaults(func=cmd_validate_first_five)

    p_ann = sub.add_parser(
        "annotate-segmentation",
        help="Write an annotated MP4 showing detected move segments",
    )
    p_ann.add_argument("--recorder", required=True, help="Recorder name (e.g. soham)")
    p_ann.add_argument("--scramble", required=True, help="Scramble id (e.g. 0001)")
    p_ann.add_argument("--fps", type=float, default=30.0, help="Target FPS for analysis/output")
    p_ann.add_argument(
        "--crop",
        default="",
        help="Optional crop box 'x0,y0,x1,y1' in pixels, or 'auto-top' for a top-region preset",
    )
    p_ann.add_argument(
        "--clap-time",
        type=float,
        default=0.0,
        help="If provided (>0), force clap time (seconds) instead of auto-detect",
    )
    p_ann.add_argument(
        "--clap-min-time",
        type=float,
        default=4.0,
        help="Ignore first N seconds when detecting the clap (helps avoid face-presentation motion)",
    )
    p_ann.add_argument(
        "--start-after-clap",
        type=float,
        default=1.0,
        help="Seconds after clap to start move segmentation",
    )
    p_ann.add_argument(
        "--out",
        default="",
        help="Output path (default: data/validation/<recorder>/<scramble>_annotated.mp4)",
    )
    p_ann.add_argument("--max-seconds", type=float, default=0.0, help="Optional cap for quicker debug")
    p_ann.set_defaults(func=cmd_annotate_segmentation)

    p_man = sub.add_parser(
        "manual-segment",
        help="Manual segmentation UI (hold Space) to mark move intervals",
    )
    p_man.add_argument("--recorder", required=True, help="Recorder name (e.g. soham)")
    p_man.add_argument("--scramble", required=True, help="Scramble id (e.g. 0001)")
    p_man.add_argument("--fps", type=float, default=30.0, help="Playback/label FPS")
    p_man.add_argument(
        "--skip-seconds",
        type=float,
        default=15.0,
        help="Skip the first N seconds before starting labeling (default 15)",
    )
    p_man.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0=normal, 2.0=2x)",
    )
    p_man.add_argument(
        "--crop",
        default="none",
        help="Crop box 'x0,y0,x1,y1', 'auto-top', or 'none' (default: none)",
    )
    p_man.add_argument("--max-seconds", type=float, default=0.0, help="Optional cap for quicker labeling")
    p_man.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: data/validation/<recorder>/<scramble>_manual_segments.json)",
    )
    p_man.set_defaults(func=cmd_manual_segment)

    p_am = sub.add_parser(
        "annotate-manual",
        help="Write an annotated MP4 using manual segments + known move list",
    )
    p_am.add_argument("--recorder", required=True, help="Recorder name (e.g. soham)")
    p_am.add_argument("--scramble", required=True, help="Scramble id (e.g. 0001)")
    p_am.add_argument(
        "--manual-json",
        default="",
        help="Path to manual segments JSON (default: data/validation/<recorder>/<scramble>_manual_segments.json)",
    )
    p_am.add_argument(
        "--crop",
        default="none",
        help="Crop box 'x0,y0,x1,y1', 'auto-top', or 'none' (default: none)",
    )
    p_am.add_argument(
        "--out",
        default="",
        help="Output MP4 path (default: data/validation/<recorder>/<scramble>_manual_annotated.mp4)",
    )
    p_am.set_defaults(func=cmd_annotate_manual)

    return parser


def cmd_annotate_segmentation(args: argparse.Namespace) -> int:
    sid = _resolve_scramble_id(args.scramble)
    meta_path = RAW_ROOT / args.recorder / sid / "metadata.json"
    if not meta_path.exists():
        print(f"No metadata at {meta_path}")
        return 1
    meta = json.loads(meta_path.read_text())
    video_path = RAW_ROOT / args.recorder / sid / meta["video_filename"]
    if not video_path.exists():
        print(f"No video at {video_path}")
        return 1

    out_path = Path(args.out) if args.out else (REPO_ROOT / "data" / "validation" / args.recorder / f"{sid}_annotated.mp4")
    crop = None
    if args.crop:
        if args.crop.strip().lower() == "auto-top":
            crop = _auto_crop_top_region(video_path)
            print(f"Using auto crop: {crop}  (x0,y0,x1,y1)")
        else:
            try:
                x0, y0, x1, y1 = (int(p.strip()) for p in args.crop.split(","))
                crop = (x0, y0, x1, y1)
            except Exception:
                print("Invalid --crop. Expected 'x0,y0,x1,y1' (integers) or 'auto-top'.")
                return 1
    cfg = AnnotateConfig(
        target_fps=args.fps,
        max_seconds=(args.max_seconds if args.max_seconds > 0 else None),
        clap_min_time_s=args.clap_min_time,
        start_after_clap_s=args.start_after_clap,
        crop=crop,
    )

    if args.clap_time and args.clap_time > 0:
        # Hacky but effective: we shift the min-time way past the forced clap and rely on
        # start_after_clap to drive segmentation start in the annotator.
        # The annotator itself uses detect_clap_frame; to force the clap exactly, we
        # pre-trim by setting clap_min_time_s to that time and making search window small.
        cfg = AnnotateConfig(
            target_fps=cfg.target_fps,
            max_seconds=cfg.max_seconds,
            start_after_clap_s=cfg.start_after_clap_s,
            clap_min_time_s=args.clap_time,
            settle_s=cfg.settle_s,
            crop=cfg.crop,
            border_px=cfg.border_px,
        )
    info = annotate_segmentation_video(
        video_path=video_path,
        out_path=out_path,
        expected_moves=int(meta.get("n_moves", 25)),
        forced_clap_time_s=(args.clap_time if args.clap_time and args.clap_time > 0 else None),
        cfg=cfg,
    )
    print(f"Wrote {out_path}")
    print(
        f"fps_used~{info['fps_used']:.1f}, clap_frame={info['clap_frame']}, "
        f"start_frame={info['start_frame']}, end_frame={info.get('end_frame')}, detected_moves={info['detected_moves']}"
    )
    return 0


def cmd_manual_segment(args: argparse.Namespace) -> int:
    sid = _resolve_scramble_id(args.scramble)
    meta_path = RAW_ROOT / args.recorder / sid / "metadata.json"
    if not meta_path.exists():
        print(f"No metadata at {meta_path}")
        return 1
    meta = json.loads(meta_path.read_text())
    video_path = RAW_ROOT / args.recorder / sid / meta["video_filename"]
    if not video_path.exists():
        print(f"No video at {video_path}")
        return 1

    crop = None
    crop_arg = (args.crop or "").strip().lower()
    if crop_arg in ("", "none", "full"):
        crop = None
    elif crop_arg == "auto-top":
        crop = _auto_crop_top_region(video_path)
        print(f"Using auto crop: {crop}  (x0,y0,x1,y1)")
    else:
        try:
            x0, y0, x1, y1 = (int(p.strip()) for p in args.crop.split(","))
            crop = (x0, y0, x1, y1)
        except Exception:
            print("Invalid --crop. Expected 'x0,y0,x1,y1' (integers), 'auto-top', or 'none'.")
            return 1

    out_json = (
        Path(args.out)
        if args.out
        else (REPO_ROOT / "data" / "validation" / args.recorder / f"{sid}_manual_segments.json")
    )
    run_manual_segmenter(
        video_path=video_path,
        out_json=out_json,
        target_fps=args.fps,
        skip_seconds=args.skip_seconds,
        speed=args.speed,
        max_seconds=(args.max_seconds if args.max_seconds > 0 else None),
        crop=crop,
    )
    print(f"Wrote {out_json}")
    return 0


def cmd_annotate_manual(args: argparse.Namespace) -> int:
    sid = _resolve_scramble_id(args.scramble)
    meta_path = RAW_ROOT / args.recorder / sid / "metadata.json"
    if not meta_path.exists():
        print(f"No metadata at {meta_path}")
        return 1
    meta = json.loads(meta_path.read_text())
    moves = list(meta.get("moves", []))

    manual_json = (
        Path(args.manual_json)
        if args.manual_json
        else (REPO_ROOT / "data" / "validation" / args.recorder / f"{sid}_manual_segments.json")
    )
    if not manual_json.exists():
        print(f"No manual segments JSON at {manual_json}")
        return 1

    video_path = RAW_ROOT / args.recorder / sid / meta["video_filename"]
    crop = None
    crop_arg = (args.crop or "").strip().lower()
    if crop_arg in ("", "none", "full"):
        crop = None
    elif crop_arg == "auto-top":
        crop = _auto_crop_top_region(video_path)
        print(f"Using auto crop: {crop}  (x0,y0,x1,y1)")
    else:
        try:
            x0, y0, x1, y1 = (int(p.strip()) for p in args.crop.split(","))
            crop = (x0, y0, x1, y1)
        except Exception:
            print("Invalid --crop. Expected 'x0,y0,x1,y1' (integers), 'auto-top', or 'none'.")
            return 1

    out_mp4 = (
        Path(args.out)
        if args.out
        else (REPO_ROOT / "data" / "validation" / args.recorder / f"{sid}_manual_annotated.mp4")
    )
    annotate_with_manual_segments(manual_json=manual_json, moves=moves, out_path=out_mp4, crop=crop)
    print(f"Wrote {out_mp4}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
