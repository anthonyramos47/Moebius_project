#!/usr/bin/env python
"""
Visual triage of the B-spline inputs: look at one surface, press Keep or Skip,
move to the next.

    python scripts/triage_bsplines.py            # resume where you left off
    python scripts/triage_bsplines.py --report   # just print the current lists
    python scripts/triage_bsplines.py --all      # revisit decided surfaces too

The decision for each surface is written to data/bsplines_triage.json the moment
you click, so closing the tab never loses work and a later run resumes at the
first undecided surface. Nothing is deleted: this records the two lists, and
acting on them is a separate step.

Buttons appear in the viewer's panel; the running status is printed to this
terminal, because kayviz captures the panel layout once when the app is
registered and never refreshes it, so a button's label cannot carry a counter
that changes. The mesh is re-registered under one constant name, which replaces
it in place and keeps whatever colour map you picked as you move along.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kayviz as kv
from moebius.utils.bsplines import (read_bspline_json, sample_bspline_surface,
                                    bspline_curvatures)

MESH = "bspline"            # constant name, so each surface replaces the last


# ── state ─────────────────────────────────────────────────────────────────────

class Triage:
    def __init__(self, names, bspline_dir: Path, state_path: Path,
                 samples: int, only_undecided: bool):
        self.names    = names
        self.dir      = bspline_dir
        self.path     = state_path
        self.samples  = samples
        self.decided  = self._load()
        self.order    = [i for i, n in enumerate(names)
                         if not (only_undecided and n in self.decided)]
        if not self.order:
            self.order = list(range(len(names)))
        self.pos      = 0
        self.history  = []          # names decided this session, for undo

    # persistence -------------------------------------------------------------
    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as e:
            print(f"[triage] could not read {self.path} ({e}); starting empty")
            return {}
        d = {n: "keep" for n in raw.get("keep", [])}
        d.update({n: "skip" for n in raw.get("skip", [])})
        return d

    def save(self) -> None:
        keep = [n for n in self.names if self.decided.get(n) == "keep"]
        skip = [n for n in self.names if self.decided.get(n) == "skip"]
        payload = {
            "_comment": ("Visual triage of data/bsplines, written by "
                         "scripts/triage_bsplines.py. 'skip' is the list "
                         "proposed for removal; nothing is deleted here."),
            "keep": keep,
            "skip": skip,
            "undecided": [n for n in self.names if n not in self.decided],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, self.path)      # atomic, so a crash cannot truncate it

    # navigation --------------------------------------------------------------
    @property
    def index(self) -> int | None:
        return self.order[self.pos] if 0 <= self.pos < len(self.order) else None

    def status(self) -> str:
        keep = sum(v == "keep" for v in self.decided.values())
        skip = sum(v == "skip" for v in self.decided.values())
        left = len(self.names) - len(self.decided)
        return (f"keep {keep}   skip {skip}   undecided {left}"
                f"   (of {len(self.names)})")

    def decide(self, verdict: str) -> None:
        i = self.index
        if i is None:
            return
        name = self.names[i]
        self.decided[name] = verdict
        self.history.append(name)
        self.save()
        print(f"  {verdict.upper():5s} [{i}] {name}        {self.status()}", flush=True)
        self.advance(+1)

    def undo(self) -> None:
        if not self.history:
            print("  nothing to undo", flush=True)
            return
        name = self.history.pop()
        self.decided.pop(name, None)
        self.save()
        # step back to that surface
        i = self.names.index(name)
        if i in self.order:
            self.pos = self.order.index(i)
        print(f"  UNDO  [{i}] {name}        {self.status()}", flush=True)
        self.render()

    def advance(self, step: int) -> None:
        self.pos += step
        if self.pos >= len(self.order):
            print(f"\n[triage] end of the list. {self.status()}")
            print(f"[triage] written to {self.path}")
            print("[triage] close the tab to finish, or press Back to revisit.")
            self.pos = len(self.order) - 1
            return
        self.pos = max(0, self.pos)
        self.render()

    # rendering ---------------------------------------------------------------
    def render(self) -> None:
        i = self.index
        if i is None:
            return
        name = self.names[i]
        shown = self.decided.get(name)
        mark  = f"  [already {shown}]" if shown else ""
        print(f"\n[{self.pos + 1}/{len(self.order)}] index {i}: {name}{mark}", flush=True)

        try:
            bsp = read_bspline_json(str(self.dir / f"{name}.json"))
            u = np.linspace(0, 1, self.samples)
            V, F = sample_bspline_surface(bsp, u, u)
            K, H, n = bspline_curvatures(bsp, u, u)
        except Exception as e:
            print(f"  cannot evaluate this surface: {type(e).__name__}: {e}")
            print("  (nothing to display — Skip is probably the answer)", flush=True)
            return

        h = H.ravel()
        flip = bool(h.min() * h.max() < 0)
        print(f"  degree {bsp.order(0)-1}x{bsp.order(1)-1}, "
              f"control points {bsp.controlpoints.shape[:2]}")
        print(f"  H in [{h.min():+.4f}, {h.max():+.4f}]   |H| min {np.abs(h).min():.4e}"
              f"   K in [{K.min():+.3f}, {K.max():+.3f}]")
        print(f"  {'H CHANGES SIGN' if flip else 'H is one-signed'}", flush=True)

        kv.register_surface_mesh(MESH, V, F)
        kv.add_scalar_quantity(MESH, "H", h, defined_on="vertices")
        kv.add_scalar_quantity(MESH, "|H|", np.abs(h), defined_on="vertices")
        kv.add_scalar_quantity(MESH, "K", K.ravel(), defined_on="vertices")
        kv.add_scalar_quantity(MESH, "r = 1/H",
                               1.0 / np.where(np.abs(h) > 1e-9, h, np.nan),
                               defined_on="vertices")
        kv.add_vector_quantity(MESH, "normal", n.reshape(-1, 3), defined_on="vertices")


# ── gui ───────────────────────────────────────────────────────────────────────

def build_gui(t: Triage):
    def gui():
        # Button labels are fixed: the panel schema is captured once, so these
        # cannot show a counter. Side effects only ever run inside a branch,
        # because the body also executes during that schema capture.
        with kv.imgui.header("Triage"):
            if kv.imgui.button("Keep"):
                t.decide("keep")
            if kv.imgui.button("Skip"):
                t.decide("skip")
        with kv.imgui.header("Navigate"):
            if kv.imgui.button("Back"):
                t.undo()
            if kv.imgui.button("Previous"):
                t.advance(-1)
            if kv.imgui.button("Next (no decision)"):
                t.advance(+1)
            if kv.imgui.button("Where am I"):
                print(f"  {t.status()}", flush=True)
                t.render()
    return gui


# ── main ──────────────────────────────────────────────────────────────────────

DEFAULT_DIR = ROOT / "data" / "bsplines"

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="directory of B-spline json files")
    ap.add_argument("--state", type=Path, default=ROOT / "data" / "bsplines_triage.json",
                    help="where the keep/skip lists are stored")
    ap.add_argument("--samples", type=int, default=40,
                    help="grid resolution for display (default 40)")
    ap.add_argument("--all", action="store_true",
                    help="revisit surfaces that already have a decision")
    ap.add_argument("--report", action="store_true",
                    help="print the stored lists and exit")
    args = ap.parse_args()

    names = sorted(p.stem for p in args.dir.glob("*.json"))
    if not names:
        print(f"[triage] no json files in {args.dir}")
        return 1

    t = Triage(names, args.dir, args.state, args.samples,
               only_undecided=not args.all)

    if args.report:
        keep = [n for n in names if t.decided.get(n) == "keep"]
        skip = [n for n in names if t.decided.get(n) == "skip"]
        rest = [n for n in names if n not in t.decided]
        print(f"{args.state}\n  {t.status()}")
        for label, lst in (("keep", keep), ("skip", skip), ("undecided", rest)):
            print(f"\n{label} ({len(lst)}):")
            for n in lst:
                print(f"  {names.index(n):3d}  {n}")
        return 0

    print(f"[triage] {len(names)} surfaces in {args.dir}")
    print(f"[triage] {t.status()}")
    print(f"[triage] decisions are saved to {args.state} on every click")
    print("[triage] use Keep / Skip in the viewer panel; Back undoes the last one")

    kv.init()
    kv.set_user_callback(build_gui(t), app_name="triage", on_load=t.render)
    kv.show()

    print(f"\n[triage] done. {t.status()}")
    print(f"[triage] lists are in {args.state}; nothing was deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
