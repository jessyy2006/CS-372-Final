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

from src.data.recording_utils import (
    ingest_recording,
    list_recorded,
    load_scramble,
    next_pending,
)
from src.data.scramble_generator import generate_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAMBLES_DIR = REPO_ROOT / "data" / "scrambles"
RAW_ROOT = REPO_ROOT / "data" / "raw"


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
