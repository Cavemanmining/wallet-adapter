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

---

# Phase 4 — the Descent

Added after Phase 2, in the same handoff style: a dependency-free Python
package, `lucifer_descent/`, implementing the endgame node web from World
Bible section 03, with the reconnect rule from section 07. It imports
`lucifer_gen` for the map probe and the PNG writer and changes nothing in it.

## What it contains

| Module | Purpose |
| --- | --- |
| `contracts.py` | Node states, the full transition table as the single source of truth, which events consume a Sigil, and the judgement calls the spec left open, written down rather than buried. |
| `web.py` | Spider-web generation from a profile seed. Planarity and tier-equals-distance hold by construction; mechanics, Pinnacle arenas and glyph nodes are assigned on tier 15. |
| `engine.py` | The rules engine: portal opening with every refusal case, the 80 percent elite rule that never clears a boss map, propagation on clear, passive points, fragments, Pinnacle unlock, and an append-only ledger that must replay to identical state. |
| `sigils.py` | Deterministic minting, the sustain drop table, and the enumerator the 170HX would use to pre-roll layouts. |
| `store.py` | One interface, memory and SQLite backends, atomic saves, optimistic revision to refuse stale writes, seeds stored as hex because SQLite integers are signed. |
| `render.py`, `cli.py` | The table view as a PNG, and commands for the whole loop, including `check` to audit a stored profile. |
| `validate.py` | Structural, state-consistency and replay checks, each proven to fail on an injected defect before it is allowed to pass. |

## Running it

```
cd lucifer-phase2
python3 -m pytest tests -q
python3 -m lucifer_descent.cli gate --profiles 300 --start-seed 1
python3 -m lucifer_descent.cli new    --profile caveman --seed 0x5EED
python3 -m lucifer_descent.cli grant  --profile caveman --tier 3 --count 2
python3 -m lucifer_descent.cli show   --profile caveman
python3 -m lucifer_descent.cli render --profile caveman --out web.png
```

## Verified results

Every number was produced by running the commands above.

```
pytest (whole repository)                700 passed in 90s   (lucifer_gen 385, unchanged)
gate  300 profiles, 200 steps each       300/300 clean, 99,300 nodes checked,
                                         183,819 ledger entries replayed, exit 0
cross-process determinism                identical web and ledger hashes under
                                         three different PYTHONHASHSEED values
independent exploit probe                12 attacks, 0 succeeded
```

A real end-to-end portal open through the CLI with the actual map generator
took 0.14 s, which is well inside the 300 ms budget the spec sets for a
pre-rolled Tier 15 portal even before any pre-rolling.

## Read this before trusting the engine

An adversarial review found nine real exploits in the first version. The
serious ones:

- **One Sigil could be spent three times.** The stash did not remember what it
  had already consumed. It now refuses any Sigil id that is in the stash,
  funding the live instance, or anywhere in the ledger.
- **Forged ledgers replayed clean.** Replay compared entries to themselves. It
  now starts from genesis and requires every entry to be exactly what the live
  engine would have written next, including the propagation batch after a
  clear, the probe facts on an open, and the closing event's Sigil.
- **The live instance was trusted over the ledger.** Editing a saved instance
  to claim no boss let elite kills clear a boss map. The open entry now records
  the probe's facts and the engine refuses an instance that disagrees.
- **A swapped web split the rules.** The engine snapshots nodes and adjacency
  at construction, and the CLI refuses a profile whose stored web is not what
  its seed generates.
- **A stale save could refund a Sigil.** Saves carry a revision and refuse to
  overwrite a store that has moved on.
- **A failed commit wedged the SQLite connection.** Commit is inside the
  transaction try, with rollback on failure.

If you port only part of this, port `engine.replay` and `validate.check_state`
intact, or the ledger stops being evidence of anything.

## Known limits, carried forward honestly

1. **Store schema is v2 and refuses v1 files.** v1 ledgers lack the probe facts
   that replay now requires, so there is nothing to migrate them from.
2. **A deleted final ledger row is undetectable** when it leaves a consistent
   earlier state. No mint counter is persisted, so it looks like restoring a
   backup. Persisting the counter would close this.
3. **`Sigil` accepts a float or bool tier at the contracts level.** The store
   refuses it; tightening the dataclass lives in contracts and was left alone.
4. **The engine's elite counter has no cap on boss maps.** A shipped test pins
   that as accepted behaviour; it cannot affect the clear rule.
5. **Small webs (`--rings` below 15) fail the tier-15 Pinnacle rule** in
   `check`, by design. They are for quick tests only.
6. **The Firebase mirror and the in-game table UI are not here.** Both need
   the machine. The SQLite store is the local side of the spec's "hashed in
   SQLite and mirrored to Firebase".
