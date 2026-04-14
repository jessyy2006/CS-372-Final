"""Ingestion utilities for scramble recordings.

Team members film scrambles on a phone, then run the ``ingest_recording`` CLI
(see ``src/data/cli.py``) which moves the video file under ``data/raw/`` and
writes a ``metadata.json`` alongside it tying the video to its known moves.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.data.scramble_generator import Scramble

VIDEO_EXTS: tuple[str, ...] = (".mov", ".mp4", ".m4v")


@dataclass(frozen=True)
class RecordingMetadata:
    scramble_id: str
    recorder: str
    video_filename: str
    moves: tuple[str, ...]
    n_moves: int
    seed: int
    ingested_at: str
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_scramble(scrambles_dir: Path, scramble_id: str) -> Scramble:
    path = scrambles_dir / f"{scramble_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No scramble file at {path}")
    return Scramble.from_json(path.read_text())


def ingest_recording(
    source_video: Path,
    scramble: Scramble,
    recorder: str,
    raw_root: Path,
    notes: str = "",
    move: bool = False,
) -> Path:
    """Place ``source_video`` under ``raw_root/{recorder}/{scramble_id}/`` and
    write its metadata.json. Returns the target directory.

    Defaults to copying rather than moving so an interrupted run leaves the
    original intact. Pass ``move=True`` once the workflow is trusted.
    """
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    if source_video.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(
            f"Unexpected extension {source_video.suffix}; expected one of {VIDEO_EXTS}"
        )

    target_dir = raw_root / recorder / scramble.scramble_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_video = target_dir / f"video{source_video.suffix.lower()}"

    if target_video.exists():
        raise FileExistsError(
            f"Recording already exists at {target_video}; refusing to overwrite"
        )

    if move:
        shutil.move(str(source_video), target_video)
    else:
        shutil.copy2(source_video, target_video)

    meta = RecordingMetadata(
        scramble_id=scramble.scramble_id,
        recorder=recorder,
        video_filename=target_video.name,
        moves=scramble.moves,
        n_moves=scramble.n_moves,
        seed=scramble.seed,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )
    (target_dir / "metadata.json").write_text(meta.to_json())
    return target_dir


def list_recorded(raw_root: Path) -> set[str]:
    """Return the set of scramble_ids already recorded under ``raw_root``."""
    recorded: set[str] = set()
    if not raw_root.exists():
        return recorded
    for recorder_dir in raw_root.iterdir():
        if not recorder_dir.is_dir():
            continue
        for scramble_dir in recorder_dir.iterdir():
            if (scramble_dir / "metadata.json").exists():
                recorded.add(scramble_dir.name)
    return recorded


def next_pending(scrambles_dir: Path, raw_root: Path) -> list[str]:
    """Scramble ids that have a JSON but no recording yet, ordered by id."""
    all_ids = sorted(p.stem for p in scrambles_dir.glob("*.json"))
    recorded = list_recorded(raw_root)
    return [sid for sid in all_ids if sid not in recorded]
