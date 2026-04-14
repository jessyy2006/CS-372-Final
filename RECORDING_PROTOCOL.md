# Recording Protocol

Follow this checklist for every session. Auto-labeling is only reliable if conditions stay consistent between scrambles.

## One-time setup (per recorder, per session)

1. **Tripod** — phone mounted at ~45° angle, ~40 cm from the cube. Lens centered on cube's resting position.
2. **Lighting** — single white LED ring light or lamp pointed at the mat. Close blinds (no sun drift over the session).
3. **Mat** — plain dark surface (solid, non-patterned). Tape a small cross on the mat to mark where the cube starts each scramble.
4. **Cube** — same physical 3×3 all session. Matte stickers preferred over glossy.
5. **Camera settings** — iPhone Camera → 1080p @ 60fps. **Tap-and-hold** on the cube to lock AE/AF. Turn off flash.
6. **Timer** — open a large on-screen stopwatch beside you (not in frame) so you can time move pacing by feel.

## Per-scramble checklist (repeat 200×)

### Before recording
- Solve the cube. Place it on the mat cross in a neutral orientation (white on top, green facing you).
- In terminal: `python -m src.data.cli show 0001` — reads the moves for the next scramble. Note the id.
- Take a breath. Hands out of frame.

### Start the recording
1. Tap record.
2. **Initial state pan (2 s)** — pick up the cube and slowly rotate it to show all 6 faces to the camera, roughly 0.3 s per face. Set it back down on the cross.
3. **Clap frame** — with both hands visible, snap your fingers once above the cube. This creates a bright motion spike we use as a sync anchor.
4. Rest for ~1 second. Hands on the mat, cube centered and still.
5. **Execute the scramble** at a slightly-slow pace (~1 move per second). Pause very briefly between moves — not a full stop, just a controlled tempo.
    - Read the moves from your notes; do not look up.
    - **No cube rotations** (x / y / z). Moves are faces only: R L U D F B and their primes / 2's.
    - If you make a mistake: tap stop, discard, restart the whole scramble.
6. **Settle (2 s)** — when the last move is done, hold still, hands back on the mat. The trailing stillness is the end marker for segmentation.
7. Stop the recording.

### After recording
1. AirDrop / transfer the clip to the laptop. Put it somewhere you remember (e.g. `~/Movies/todo/`).
2. Ingest it:
   ```bash
   python -m src.data.cli ingest --scramble 0001 --recorder alice ~/Movies/todo/IMG_1234.MOV
   ```
   Use `--move` once you trust the pipeline so you don't leave duplicates.
3. Verify: `python -m src.data.cli list-pending` should show one fewer pending.

## Things that invalidate a take (re-record)

- Hand partially occludes the whole cube for more than ~0.5 s during a move.
- You performed a cube rotation (x / y / z) by accident.
- Lighting visibly changes (someone opened a curtain, etc.) mid-take.
- Camera got bumped.
- You lost count / skipped / doubled a move.

## First-five validation gate

Before recording all 200, film scrambles `0000`–`0004` and run motion segmentation (Day 2 work) on them. If ≥1 of 5 misaligns, either:

- **Option A (preferred):** relax pacing — re-record with a deliberate 500 ms pause between each move. Trades realism for label cleanliness, fine because our final eval set is un-paced real solves, not these scripted scrambles.
- **Option B:** fix the recording conditions (lighting, angle, cube contrast) and retry.

## Target pacing math

200 scrambles × 25 moves × 1.0 s/move ≈ 83 min of pure move time. With setup / reset / transfer: **budget 4–6 hours total** across the team.

## Partner split

- **Partner A** primary recorder for scrambles 0000–0099.
- **Partner B** primary recorder for scrambles 0100–0199.
- Recorder name goes into ingestion CLI (`--recorder alice` vs `--recorder bob`) — this populates `metadata.json` and keeps us honest about who-recorded-what for the contributions rubric item.
