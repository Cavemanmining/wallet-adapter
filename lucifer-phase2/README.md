# Lucifer Phase 2 — greybox map generator

A complete, dependency-free Python implementation of stages 1 to 6 of the
Lucifer map generator described in the project's World Bible, together with
the Phase 2 validation gate, a command-line tool, and a PNG renderer.

**This directory is a handoff artifact, not part of wallet-adapter.** It was
built in a cloud session while the Jarvis machine was unavailable, and is
committed here only because that session's container is ephemeral and this was
the one repository it could push to. Move it into the Lucifer repository and
delete it from here.

## Why it exists

The Phase 2 Python lane stalled on Jarvis partway through. Everything in this
package was written against the spec so it can be reconciled with, or dropped
in place of, whatever that lane produced.

## What it contains

| Module | Stage | Purpose |
| --- | --- | --- |
| `contracts.py` | — | Shared types, the side and transform algebra, and the rule for when two tiles may abut. Every other module builds against this. |
| `seed.py` | — | The 64-bit seed field layout, with each stage drawing from its own stream so editing one field leaves the others byte-identical. |
| `shapes.py` | 2 | The six macro shapes and their anchors, rotation, mirroring and end swapping. |
| `template.py` | 1 | Template loading and validation. |
| `route.py` | 2 | Node placement with jitter and separation, then A* edge routing. |
| `tiles.py`, `rooms.py` | 3-4 | Greybox tile database with proven coverage, and the room library. |
| `translate.py` | 3 | Corridors and socket-matched rooms for dungeons; Bezier splines with offset cliffs and banks for outdoor. |
| `tileize.py` | 4 | Signature matching with rotation and flip, filler, and the hero budget. |
| `setpieces.py` | 5 | Fixed interiors snapped to the arriving corridor, plus the exit and boss-approach tells. |
| `spawn.py`, `layout.py` | 6 | Pack placement, then the client layout description and its hash. |
| `validate.py` | gate | Edge-aware navmesh islands, seam checks under two independent models, and the stage 4 invariants. |
| `png.py`, `render.py` | — | A PNG writer and map renderer using only the standard library. |
| `pipeline.py`, `cli.py` | — | `generate()` and the command-line entry points. |

## Running it

No third-party packages are required to run the generator. `pytest` is needed
only for the test suite.

```
cd lucifer-phase2
python3 -m pytest tests -q

python3 -m lucifer_gen.cli validate --template crypt           --seeds 1000 --start-seed 1
python3 -m lucifer_gen.cli validate --template ashen_ramparts  --seeds 1000 --start-seed 1
python3 -m lucifer_gen.cli render   --template crypt --seed 0xA11CE  --out crypt.png
python3 -m lucifer_gen.cli graph    --template crypt --seed 0xA11CE  --out graph.png
python3 -m lucifer_gen.cli describe --template crypt --seed 0xA11CE
```

The CLI must run from this directory, or with `PYTHONPATH` pointing at it.
There is deliberately no `setup.py`, so packaging is the receiving repository's
choice.

## Verified results

Every number below was produced by running the commands above, not estimated.

```
pytest                                385 passed in 114s
validate crypt           1000 seeds   1000 clean, islands 0, seams 0,  68s, exit 0
validate ashen_ramparts  1000 seeds   1000 clean, islands 0, seams 0,  93s, exit 0
```

Cross-process determinism holds: the same seed yields an identical layout hash
and a byte-identical description and PNG across separate processes with
randomised hash seeds.

## Read this before trusting the gate

An adversarial review found that the first version of the gate **could not
fail**. Both checks tested only whether neighbouring cells were adjacent and
walkable, never whether the seam between them was passable, so a map with
24,751 of 24,851 cells sealed off still reported zero islands and zero seams.

That is fixed here, and the fix is the reason the gate is worth anything:

- The flood fill is edge-aware. A step counts only when both tiles leave the
  shared edge open with overlapping connection slots.
- Seams are judged by two genuinely independent models. The second derives slot
  positions geometrically rather than reusing the same helper, and a
  disagreement between the two is reported as loudly as a failure.
- The stage 4 invariant check, which previously existed but was never called,
  now runs on every seed.

If you port only part of this package, port `validate.py` intact or you will
inherit a gate that reports success unconditionally.

## Known limits, carried forward honestly

1. **Tile ids are one byte.** `Placement.packed` caps the database at 256 tiles;
   the greybox set uses 141. Going beyond that needs a contracts change and a
   wider cell encoding.
2. **The chosen landmark is not in the client description.** It is reachable on
   `GeneratedMap.markers`, but the description keeps exactly the ten fields the
   spec names. Adding an eleventh key is a one-line change once the spec allows.
3. **Small grids can fail to route.** Templates admit grids from 8 cells, but a
   crypt-shaped template at 16 cells cannot satisfy the 6-cell node separation
   for some seeds and raises. The shipped templates use 48 and are unaffected.
4. **Spawn pack names are placeholders** in `spawn.py`, not content data.
5. **Greybox meshes are named, not modelled.** Tiles carry mesh references such
   as `greybox/dungeon/cross_statue`. Swapping in the Blender library is a data
   change; no generator code should need to move.
