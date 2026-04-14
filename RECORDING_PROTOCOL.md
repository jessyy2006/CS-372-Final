# Recording Protocol

Follow this checklist for every session. Auto-labeling is only reliable if conditions stay consistent between scrambles.

**View:** high-angle bird's-eye, **tilted ~15° off vertical toward the solver**. The U (top) face dominates the frame and a sliver of the front face is visible. This is a deliberate compromise: the tilt lets the classifier tell F from B and L from R more reliably than a pure overhead view, without giving up the consistent top-down framing that makes segmentation easy.

## One-time setup (per recorder, per session)

1. **Camera rig** — phone mounted above the cube, tilted **~15° off vertical toward the solver** (i.e., the top of the phone leans away from you, lens angled slightly back at your side of the cube). Height: ~30–35 cm above the cube. The cube should appear centered in the viewfinder with all four side edges of the top face visible plus a shallow slice of the front face. A stack of books or a phone clip on a gooseneck works fine — measure the tilt with a phone level app once and mark the rig position with tape so you can rebuild it next session.
2. **Lighting** — whatever room lighting is available, but **keep it fixed for the entire session**. Once you start recording, do not turn lights on/off, open blinds, or move lamps. Avoid overhead single-point lights that cast a hard hand shadow directly on the cube — angle yourself so your hands shadow *off* to the side of the cube rather than onto it.
3. **Mat** — plain dark surface (solid, non-patterned), larger than the frame. Tape a small cross on the mat to mark where the cube starts each scramble.
4. **Cube** — same physical 3×3 all session. Matte stickers preferred over glossy (glossy creates specular glare from the ceiling light).
5. **Camera settings** — iPhone Camera → 1080p @ 60fps. **Tap-and-hold** on the cube to lock AE/AF (critical — the camera will otherwise re-expose every time a hand enters frame). Turn off flash.
6. **Timer** — open a large on-screen stopwatch beside you (not in frame) so you can time move pacing by feel.

## Per-scramble checklist (repeat 200×)

### Before recording
- Solve the cube. Place it on the mat cross in a neutral orientation (white on top, green facing you — the side of the cube closest to your body as you sit).
- In terminal: `python -m src.data.cli show 0001` — reads the moves for the next scramble. Note the id.
- Hands out of frame, palms clear.

### Start the recording
1. Tap record.
2. **Initial state capture (~6 s)** — the 15°-tilted view doesn't show all faces, so you need to present each explicitly. Pick up the cube, hold it centered in the viewfinder ~15–20 cm below the lens, and slowly present each face **flat to the lens** (perpendicular to the camera's line of sight, not to the mat) for ~1 second in this order: **U (white), D (yellow), F (green), B (blue), L (orange), R (red)**. Hold each face still for the full second — motion blur ruins color classification. Set the cube back on the cross in the neutral orientation (white up, green toward you).
3. **Clap frame** — with both hands visible in frame, snap your fingers once above the cube. This creates a bright motion spike we use as a sync anchor between the state-capture phase and the scramble phase.
4. Rest for ~1 second. Hands on the mat, cube centered and still.
5. **Execute the scramble** at a slightly-slow pace (~1 move per second). Pause very briefly between moves — not a full stop, just a controlled tempo.
    - Read the moves from your notes; do not look up.
    - **No cube rotations** (x / y / z). Moves are faces only: R L U D F B and their primes / 2's.
    - Keep the cube on or very near the mat cross. Large translations between moves degrade segmentation.
    - **Pause 500 ms between moves** — a deliberate, visible stillness. This is the single highest-leverage thing for auto-labeling accuracy; it gives segmentation clean boundaries and the classifier a clean "before/after" snapshot per move.
    - For side moves (R/L/F/B), turn the face with a clean **pinch motion** — don't cup the whole side of the cube with your palm, since cupping occludes the stickers we need to see rotate.
    - **Consistent hand-approach direction:** R from the right, L from the left, F from near your body, B from the far side, U and D from directly above. The hand-trajectory signal matters a lot at this view angle — make it unambiguous.
    - Both hands should remain visible in frame throughout — never let one hand disappear under the cube.
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

- Hand fully covers the top face for more than ~0.3 s during a move. (Top-down has no backup view — if U is occluded we see nothing.)
- You performed a cube rotation (x / y / z) by accident.
- Room lighting visibly changes (someone opened a curtain, flipped a light) mid-take.
- The cube slides significantly off the mat cross during a move.
- Camera got bumped.
- You lost count / skipped / doubled a move.

## First-five validation gate

Before recording all 200, film scrambles `0000`–`0004` and run motion segmentation (Day 2 work) on them. If ≥1 of 5 misaligns, either:

- **Option A (preferred):** relax pacing — re-record with a deliberate 500 ms pause between each move. Trades realism for label cleanliness, fine because our final eval set is un-paced real solves, not these scripted scrambles.
- **Option B:** fix the recording conditions (lighting, hand shadows, cube contrast vs mat) and retry.

Additionally, verify on the first five that all six faces are clearly visible during the initial-state-capture phase — if not, re-record the state phase specifically, since scramble-state detection depends on it.

## Target pacing math

200 scrambles × 25 moves × 1.0 s/move ≈ 83 min of pure move time. Add ~10 s/scramble for the initial face-presentation + clap + reset = ~33 min overhead. With transfer time: **budget 4–6 hours total** across the team.

## Partner split

- **Partner A** primary recorder for scrambles 0000–0099.
- **Partner B** primary recorder for scrambles 0100–0199.
- Recorder name goes into ingestion CLI (`--recorder alice` vs `--recorder bob`) — this populates `metadata.json` and keeps us honest about who-recorded-what for the contributions rubric item.
