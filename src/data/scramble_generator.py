"""WCA-style scramble generation for 3x3 Rubik's cubes.

Produces move sequences over the 18-move alphabet {R, R', R2, L, L', L2, U, U',
U2, D, D', D2, F, F', F2, B, B', B2} with two constraints:

  1. No consecutive moves on the same face (R then R' is redundant).
  2. No three consecutive moves on the same axis (R L R, all x-axis, is
     suboptimal and visually ambiguous for segmentation).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

FACES: tuple[str, ...] = ("R", "L", "U", "D", "F", "B")
SUFFIXES: tuple[str, ...] = ("", "'", "2")
MOVES: tuple[str, ...] = tuple(f + s for f in FACES for s in SUFFIXES)

# Each face belongs to one of three parallel axes. Two moves on the same axis
# commute, so three-in-a-row would equivalently be two moves (reducible).
FACE_TO_AXIS: dict[str, str] = {
    "R": "x", "L": "x",
    "U": "y", "D": "y",
    "F": "z", "B": "z",
}


@dataclass(frozen=True)
class Scramble:
    """One generated scramble with provenance metadata."""

    scramble_id: str
    moves: tuple[str, ...]
    seed: int
    n_moves: int
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Scramble":
        obj = json.loads(text)
        obj["moves"] = tuple(obj["moves"])
        return cls(**obj)


def generate_scramble(n_moves: int = 25, seed: int | None = None) -> tuple[str, ...]:
    """Return a WCA-style scramble of ``n_moves`` moves.

    The constraints eliminate trivially cancellable sequences so that every
    recorded move produces an observable visual change.
    """
    if n_moves <= 0:
        raise ValueError("n_moves must be positive")
    rng = random.Random(seed)
    seq: list[str] = []
    while len(seq) < n_moves:
        move = rng.choice(MOVES)
        face = move[0]
        axis = FACE_TO_AXIS[face]
        if seq:
            prev_face = seq[-1][0]
            if face == prev_face:
                continue
            # Disallow three consecutive moves on the same axis (A B A where
            # both axes match). Two-in-a-row is fine; three is reducible.
            if (
                len(seq) >= 2
                and FACE_TO_AXIS[prev_face] == axis
                and FACE_TO_AXIS[seq[-2][0]] == axis
            ):
                continue
        seq.append(move)
    return tuple(seq)


def make_scramble(scramble_id: str, n_moves: int = 25, seed: int | None = None) -> Scramble:
    """Generate a :class:`Scramble` with full provenance recorded."""
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    moves = generate_scramble(n_moves=n_moves, seed=seed)
    return Scramble(
        scramble_id=scramble_id,
        moves=moves,
        seed=seed,
        n_moves=n_moves,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def generate_batch(
    count: int,
    output_dir: Path,
    n_moves: int = 25,
    master_seed: int = 0,
    id_prefix: str = "scramble",
) -> list[Scramble]:
    """Generate ``count`` scrambles and write them under ``output_dir``.

    Each scramble's seed is derived deterministically from ``master_seed`` so
    the full batch is reproducible from a single integer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(master_seed)
    scrambles: list[Scramble] = []
    for i in range(count):
        seed = master_rng.randrange(2**31)
        scramble_id = f"{id_prefix}_{i:04d}"
        s = make_scramble(scramble_id, n_moves=n_moves, seed=seed)
        (output_dir / f"{scramble_id}.json").write_text(s.to_json())
        scrambles.append(s)
    return scrambles
