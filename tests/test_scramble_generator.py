"""Unit tests for src.data.scramble_generator.

Run from repo root: `pytest tests/test_scramble_generator.py -q`
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.scramble_generator import (
    FACE_TO_AXIS,
    MOVES,
    Scramble,
    generate_batch,
    generate_scramble,
    make_scramble,
)


def test_move_alphabet_is_18():
    assert len(MOVES) == 18
    assert len(set(MOVES)) == 18


def test_scramble_has_correct_length():
    for n in (5, 25, 40):
        seq = generate_scramble(n_moves=n, seed=42)
        assert len(seq) == n


def test_scramble_is_deterministic_given_seed():
    a = generate_scramble(n_moves=25, seed=123)
    b = generate_scramble(n_moves=25, seed=123)
    assert a == b


def test_no_consecutive_same_face():
    seq = generate_scramble(n_moves=100, seed=7)
    for prev, curr in zip(seq, seq[1:]):
        assert prev[0] != curr[0], f"consecutive same-face moves: {prev} {curr}"


def test_no_three_consecutive_same_axis():
    seq = generate_scramble(n_moves=200, seed=7)
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        axes = {FACE_TO_AXIS[a[0]], FACE_TO_AXIS[b[0]], FACE_TO_AXIS[c[0]]}
        assert len(axes) >= 2, f"three-in-a-row on same axis: {a} {b} {c}"


def test_every_move_is_in_alphabet():
    seq = generate_scramble(n_moves=300, seed=11)
    for mv in seq:
        assert mv in MOVES


def test_invalid_n_moves_raises():
    with pytest.raises(ValueError):
        generate_scramble(n_moves=0)
    with pytest.raises(ValueError):
        generate_scramble(n_moves=-1)


def test_make_scramble_roundtrips_through_json():
    s = make_scramble("scramble_test", n_moves=25, seed=99)
    restored = Scramble.from_json(s.to_json())
    assert restored == s


def test_generate_batch_writes_files(tmp_path: Path):
    batch = generate_batch(count=10, output_dir=tmp_path, n_moves=25, master_seed=0)
    assert len(batch) == 10
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 10
    # Ids are zero-padded, sequential
    assert files[0].stem == "scramble_0000"
    assert files[-1].stem == "scramble_0009"
    # Seeds should differ across scrambles even from the same master seed
    seeds = {s.seed for s in batch}
    assert len(seeds) == 10


def test_batch_is_reproducible(tmp_path: Path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a = generate_batch(count=5, output_dir=a_dir, master_seed=42)
    b = generate_batch(count=5, output_dir=b_dir, master_seed=42)
    assert [s.moves for s in a] == [s.moves for s in b]
