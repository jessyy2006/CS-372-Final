"""Unit tests for src.data.recording_utils."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.recording_utils import (
    VIDEO_EXTS,
    ingest_recording,
    list_recorded,
    load_scramble,
    next_pending,
)
from src.data.scramble_generator import generate_batch, make_scramble


def _make_fake_video(path: Path, ext: str = ".mov") -> Path:
    video = path / f"phone_clip{ext}"
    video.write_bytes(b"fake video bytes")
    return video


def test_ingest_copies_video_and_writes_metadata(tmp_path: Path):
    scramble = make_scramble("scramble_0001", n_moves=25, seed=1)
    raw_root = tmp_path / "raw"
    source = _make_fake_video(tmp_path)

    target_dir = ingest_recording(
        source_video=source,
        scramble=scramble,
        recorder="alice",
        raw_root=raw_root,
    )
    assert target_dir == raw_root / "alice" / "scramble_0001"
    assert (target_dir / "video.mov").exists()
    assert source.exists(), "copy mode should leave the source in place"
    meta = json.loads((target_dir / "metadata.json").read_text())
    assert meta["scramble_id"] == "scramble_0001"
    assert meta["recorder"] == "alice"
    assert meta["n_moves"] == 25


def test_ingest_with_move_removes_source(tmp_path: Path):
    scramble = make_scramble("scramble_0002", seed=2)
    raw_root = tmp_path / "raw"
    source = _make_fake_video(tmp_path)

    ingest_recording(source, scramble, "bob", raw_root, move=True)
    assert not source.exists()


def test_ingest_refuses_overwrite(tmp_path: Path):
    scramble = make_scramble("scramble_0003", seed=3)
    raw_root = tmp_path / "raw"
    source1 = _make_fake_video(tmp_path)
    ingest_recording(source1, scramble, "alice", raw_root)

    source2 = tmp_path / "second.mov"
    source2.write_bytes(b"another")
    with pytest.raises(FileExistsError):
        ingest_recording(source2, scramble, "alice", raw_root)


def test_ingest_rejects_unknown_extension(tmp_path: Path):
    scramble = make_scramble("scramble_0004", seed=4)
    raw_root = tmp_path / "raw"
    bogus = tmp_path / "clip.avi"
    bogus.write_bytes(b"x")
    with pytest.raises(ValueError):
        ingest_recording(bogus, scramble, "alice", raw_root)


def test_list_recorded_and_next_pending(tmp_path: Path):
    scrambles_dir = tmp_path / "scrambles"
    raw_root = tmp_path / "raw"
    batch = generate_batch(count=3, output_dir=scrambles_dir, master_seed=0)

    assert list_recorded(raw_root) == set()
    assert next_pending(scrambles_dir, raw_root) == [s.scramble_id for s in batch]

    source = _make_fake_video(tmp_path)
    ingest_recording(source, batch[0], "alice", raw_root)

    assert list_recorded(raw_root) == {batch[0].scramble_id}
    assert next_pending(scrambles_dir, raw_root) == [
        batch[1].scramble_id,
        batch[2].scramble_id,
    ]


def test_load_scramble_roundtrip(tmp_path: Path):
    scrambles_dir = tmp_path / "scrambles"
    batch = generate_batch(count=2, output_dir=scrambles_dir, master_seed=7)
    loaded = load_scramble(scrambles_dir, batch[0].scramble_id)
    assert loaded == batch[0]


def test_video_exts_covers_common_iphone_formats():
    assert ".mov" in VIDEO_EXTS
    assert ".mp4" in VIDEO_EXTS
