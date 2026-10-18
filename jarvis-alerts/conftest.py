"""Put the sibling handoff packages on the import path.

The handoff packages ship as plain directories with no install metadata,
and they import each other: every one of them needs ``lucifer_gen.seed``,
and the bots and the poke assistant reach into each other's contracts.
Each test module already inserts its own package root, which is enough to
import the package under test and nothing else, so without this the suite
only collects under a hand-built four-way ``PYTHONPATH``.

Both the path and the environment are set, because the suites also drive
the CLIs through ``subprocess``: a child interpreter inherits the
environment, never this process's ``sys.path``.

A sibling that is not part of the checkout is skipped, so each package
still stands alone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SIBLINGS = ("lucifer-phase2", "jarvis-alerts", "jarvis-bots", "jarvis-poke")

ROOTS = [str(REPO_ROOT / name) for name in SIBLINGS if (REPO_ROOT / name).is_dir()]

for _root in ROOTS:
    if _root not in sys.path:
        sys.path.insert(0, _root)

_inherited = os.environ.get("PYTHONPATH", "")
_entries = [p for p in _inherited.split(os.pathsep) if p]
_entries = ROOTS + [p for p in _entries if p not in ROOTS]
os.environ["PYTHONPATH"] = os.pathsep.join(_entries)
