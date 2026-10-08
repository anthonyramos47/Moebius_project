#!/usr/bin/env python
"""
Batch stage-1b: condition the mean curvature of every B-spline in data/bsplines
and write the result to data/bsplines_optimized.

    python scripts/optimize_bsplines.py                      # all of them
    python scripts/optimize_bsplines.py --jobs 24
    python scripts/optimize_bsplines.py --h-max-percentile 99 # also cap |H|
    python scripts/optimize_bsplines.py --report              # read the summary

Built to be left running: every surface is independent, each one's result and
log are written as it finishes, and a second run skips whatever already has an
output file. So it survives being killed, and `--force` redoes everything.

    nohup python scripts/optimize_bsplines.py > opt.log 2>&1 &
    tail -f opt.log

The curvature ceiling is on by default
--------------------------------------
The floor, |H| >= H_thresh, is satisfied most cheaply by a local dent, and on
some surfaces that is exactly what happens. With no ceiling, measured at 30x30,
max |H| goes from 30.2 to 4081 on Roof, 16.5 to 557 on Roof_L and 31.2 to 516
on last_ex_1 -- radii of order 1e-4 on a surface two units across, which is a
crease rather than a shape. Capping |H| at the 99th percentile of each
surface's own initial |H| brings those to 17.1, 6.5 and 14.5 while still
meeting the floor and still removing the sign change, so that is the default.
A percentile rather than an absolute value, because the right cap is a property
of the surface. --no-h-max restores the old behaviour.

The ceiling's weight matters as much as the ceiling. Given equal weight to the
floor (w_max=1) the two hinges compete and the floor loses: over the whole set
the surfaces still changing sign went from 3 to 6, and Complex_test_S ran away
to max |H| = 13348. At w_max=0.01 the ceiling is a preference rather than a
rival, and on the seven surfaces that showed either failure it fixes every sign
change *and* lowers the curvature -- Roof 30.2 -> 17.1, Roof_L 16.5 -> 6.6,
last_ex_1 31.2 -> 14.5, tunel_inv 42.5 -> 5.4, and Complex_test_S 13.6 -> 7.3
instead of 13348. Hence the default.

Surfaces that already satisfy the condition are not optimised
-------------------------------------------------------------
If |H| is already above the threshold everywhere and does not change sign,
there is nothing for the stage to do -- and running it anyway makes the surface
worse. The curvature residual is zero at the start, so the only active term
left is the Laplacian smoothing on the control net, which then drives the
deformation on its own until the curvature floor stops it. Measured on Tunel at
30x30: untouched with w_smooth=0, but with the default w_smooth=1e-2 it ran 14
iterations, took max |H| from 4.06 to 18.14 and ended with 3 samples *below*
the very threshold it was enforcing. So those surfaces are normalised and
written out unchanged, with status "already_ok". --no-skip-satisfied forces
them through the optimiser.

Parallelism: over surfaces, not inside one
------------------------------------------
Each surface is an independent problem, so each gets its own process, pinned to
a single BLAS thread so the workers do not fight over cores. With the jax
Jacobian a surface takes a few seconds, so this is quick either way; with
--no-jax it is the difference between minutes and hours.

Normalisation
-------------
Control points are normalised twice, both via Chakana's normalize_vertices:
once on load (read_bspline_json does it) and once after the optimisation, since
the deformation moves the bounding box. The second one is why the saved file is
directly comparable to the inputs. It also rescales curvature, so the summary
reports |H| both before and after that rescaling -- the saved surface is judged
on the second one, which is what a later run will actually see.
"""

from __future__ import annotations

# Pin every numerical library to one thread BEFORE numpy is imported, so that N
# worker processes do not each start a full thread pool and thrash the cores.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

DEFAULT_IN  = ROOT / "data" / "bsplines"
DEFAULT_OUT = ROOT / "data" / "bsplines_optimized"


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("per-surface time limit reached")


def _stats(H, thresh):
    h = np.asarray(H).ravel()
    a = np.abs(h)
    return {
        "H_min":       float(h.min()),
        "H_max":       float(h.max()),
        "absH_min":    float(a.min()),
        "absH_max":    float(a.max()),
        "absH_p99":    float(np.percentile(a, 99)),
        "sign_change": bool(h.min() * h.max() < 0),
        "below":       int((a < thresh).sum()),
        "n":           int(a.size),
    }


# ── worker ────────────────────────────────────────────────────────────────────

def run_one(job: dict) -> dict:
    """Condition one surface. Runs in its own process; never raises."""
    name = job["name"]
    out  = Path(job["out_dir"]) / f"{name}.json"
    log  = Path(job["out_dir"]) / "logs" / f"{name}.log"
    rec  = {"name": name, "status": "error", "seconds": 0.0}
    t0   = time.time()
    buf  = StringIO()

    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(job["timeout"]))

        # imported inside the worker so a spawned process sets up cleanly
        from hanan.geometry.construction import normalize_vertices
        from moebius.pipeline import (load_surface, setup_bspline_optimizer,
                                      run_bspline_optimizer)
        from moebius.utils.bsplines import (read_bspline_json, save_bspline_to_json,
                                            sample_bspline_surface, bspline_curvatures)

        thresh = job["H_thresh"]
        with redirect_stdout(buf), redirect_stderr(buf):
            state = load_surface(name, "batch",
                                 bspline_dir=Path(job["in_dir"]),
                                 out_dir=Path(job["out_dir"]),
                                 u_sample_num=job["samples"],
                                 v_sample_num=job["samples"])
            u, v = state.u_pts, state.v_pts
            V0, _ = sample_bspline_surface(state.bspline, u, v)
            _, H0, _ = bspline_curvatures(state.bspline, u, v)

            rec["degree"] = [state.bspline.order(0) - 1, state.bspline.order(1) - 1]
            rec["control_points"] = list(state.bspline.controlpoints.shape[:2])
            rec["before"] = _stats(H0, thresh)

            # A ceiling has to be scaled per surface, so it is given as a
            # percentile of this surface's own |H| rather than an absolute.
            H_max = None
            if job["h_max_percentile"] is not None:
                H_max = float(np.percentile(np.abs(np.asarray(H0).ravel()),
                                            job["h_max_percentile"]))
                if H_max <= thresh:          # degenerate: the hinges would fight
                    H_max = None
            rec["H_max"] = H_max

            # Nothing to solve, and solving anyway is harmful: see the module
            # docstring. Normalise and write it out as it is.
            b = rec["before"]
            satisfied = (not b["sign_change"]) and b["below"] == 0
            rec["already_ok"] = bool(satisfied and job["skip_satisfied"])

            if rec["already_ok"]:
                rec["iterations"] = 0
                rec["energy_final"] = rec["energy_best"] = None
            else:
                setup_bspline_optimizer(state, H_thresh=thresh, H_max=H_max,
                                        w_max=job["w_max"], w_H=job["w_H"],
                                        w_step=job["w_step"],
                                        w_smooth=job["w_smooth"],
                                        use_jax=job["use_jax"])
                run_bspline_optimizer(state, max_iter=job["max_iter"])

                opt = state.bspline_opt
                rec["iterations"] = int(getattr(opt, "it", -1) or 0)
                energies = list(getattr(opt, "energy", []) or [])
                rec["energy_final"] = float(energies[-1]) if energies else None
                rec["energy_best"]  = float(min(energies)) if energies else None

            V1, _ = sample_bspline_surface(state.bspline, u, v)
            _, H1, _ = bspline_curvatures(state.bspline, u, v)
            d = np.linalg.norm(V1 - V0, axis=1)
            rec["moved_median"] = float(np.median(d))
            rec["moved_max"]    = float(d.max())
            rec["after"] = _stats(H1, thresh)

            # ── normalise the control points again and save ──────────────────
            cp = np.asarray(state.bspline.controlpoints)
            flat = cp.reshape(-1, 3)
            span_before = float((flat.max(axis=0) - flat.min(axis=0)).max())
            state.bspline.controlpoints = normalize_vertices(
                flat, job["factor"]).reshape(cp.shape)
            flat2 = np.asarray(state.bspline.controlpoints).reshape(-1, 3)
            rec["bbox_before_norm"] = span_before
            rec["bbox_after_norm"]  = float((flat2.max(axis=0) - flat2.min(axis=0)).max())

            _, H2, _ = bspline_curvatures(state.bspline, u, v)
            rec["saved"] = _stats(H2, thresh)

            out.parent.mkdir(parents=True, exist_ok=True)
            save_bspline_to_json(state.bspline, str(out))

            # the save must be exact or the file is not the surface we measured
            back = read_bspline_json(str(out), normalize=False)
            rec["roundtrip_max_dcp"] = float(np.abs(
                np.asarray(back.controlpoints) - np.asarray(state.bspline.controlpoints)
            ).max())

        # The hinge converges *to* the threshold from below, so a converged
        # surface sits a hair under it and a strict |H| < thresh test calls
        # that a failure. Measured: surface_00002 converges at |H|min =
        # 0.9999780 x thresh and rot at 0.9994155, while CM_Strip genuinely
        # fails at 0.0331 x thresh. So the test is a relative margin, which
        # separates those cleanly, and the margin itself is reported.
        s = rec["saved"]
        a2 = np.abs(np.asarray(H2).ravel())
        rec["absH_min_over_thresh"] = float(a2.min() / thresh)
        tol = thresh * (1.0 - job["accept_tol"])
        rec["below_tol"] = int((a2 < tol).sum())
        rec["ok"] = bool(not s["sign_change"] and rec["below_tol"] == 0
                         and rec["roundtrip_max_dcp"] < 1e-10)
        rec["status"] = ("already_ok" if (rec["ok"] and rec.get("already_ok"))
                         else "ok" if rec["ok"] else "incomplete")

    except Timeout as e:
        rec["status"], rec["error"] = "timeout", str(e)
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        buf.write("\n" + traceback.format_exc())
    finally:
        signal.alarm(0)
        rec["seconds"] = round(time.time() - t0, 2)
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(buf.getvalue())
        except OSError:
            pass
    return rec


# ── driver ────────────────────────────────────────────────────────────────────

def write_summary(path: Path, meta: dict, records: dict) -> None:
    payload = dict(meta)
    done = list(records.values())
    payload["counts"] = {
        "total": meta["n_surfaces"],
        "done": len(done),
        "ok": sum(r["status"] == "ok" for r in done),
        "already_ok": sum(r["status"] == "already_ok" for r in done),
        "incomplete": sum(r["status"] == "incomplete" for r in done),
        "timeout": sum(r["status"] == "timeout" for r in done),
        "error": sum(r["status"] == "error" for r in done),
    }
    payload["results"] = [records[k] for k in sorted(records)]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)          # atomic: a kill cannot leave it truncated


def line(r: dict) -> str:
    if r["status"] in ("error", "timeout"):
        return f"  {r['status'].upper():10s} {r['name']:22s} {r.get('error','')[:60]}"
    b, s = r["before"], r["saved"]
    flag = {"ok": "ok  ", "already_ok": "as-is", "incomplete": "INC "}.get(r["status"], "????")
    return (f"  {flag} {r['name']:22s} "
            f"|H|min {b['absH_min']:.2e}->{s['absH_min']:.2e} "
            f"({r.get('absH_min_over_thresh', float('nan')):7.4f}x thr)  "
            f"short {r.get('below_tol', -1):4d} "
            f"flip {str(b['sign_change'])[0]}->{str(s['sign_change'])[0]}  "
            f"max|H| {b['absH_max']:8.2f}->{s['absH_max']:<8.2f} "
            f"moved {r['moved_max']:.1e}  {r['iterations']:3d}it {r['seconds']:6.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir",  type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--samples", type=int, default=30,
                    help="u and v samples for the optimisation grid (default 30)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default: half the cores)")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--h-thresh", type=float, default=0.05)
    ap.add_argument("--h-max-percentile", type=float, default=99.0,
                    help="cap |H| at this percentile of the surface's own initial "
                         "|H| (default 99). This is what stops the floor being "
                         "met by a crease; the module docstring has the measured "
                         "effect.")
    ap.add_argument("--no-h-max", action="store_true",
                    help="drop the curvature ceiling entirely (the old behaviour; "
                         "it creases some surfaces badly)")
    ap.add_argument("--w-max",  type=float, default=0.01,
                    help="weight of the curvature ceiling relative to the floor "
                         "(default 0.01). At 1.0 the two hinges compete and the "
                         "floor loses; see the module docstring.")
    ap.add_argument("--w-h",    type=float, default=1.0)
    ap.add_argument("--w-step", type=float, default=0.1)
    ap.add_argument("--w-smooth", type=float, default=1e-2,
                    help="Laplacian smoothing on the control net (pipeline "
                         "default 1e-2). It is the only active term once the "
                         "curvature floor is met, so it, not the floor, decides "
                         "what a converged surface looks like; 0 measured "
                         "slightly better on the one surface tested.")
    ap.add_argument("--no-jax", action="store_true",
                    help="use the finite-difference Jacobian instead of jax "
                         "(~40x slower, and less accurate)")
    ap.add_argument("--no-skip-satisfied", action="store_true",
                    help="optimise even surfaces that already meet the "
                         "condition; see the module docstring for why that is "
                         "off by default")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="longest bounding-box dimension after normalisation")
    ap.add_argument("--accept-tol", type=float, default=1e-3,
                    help="relative margin by which |H| may fall short of "
                         "H_thresh and still count as met (default 1e-3). The "
                         "hinge converges to the threshold from below, so an "
                         "exact test misreads a converged surface as a failure.")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="seconds allowed per surface (default 3600)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these surface names")
    ap.add_argument("--force", action="store_true",
                    help="redo surfaces that already have an output file")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="print the existing summary and exit")
    args = ap.parse_args()

    summary = args.out_dir / "summary.json"

    if args.report:
        if not summary.exists():
            print(f"no summary at {summary}")
            return 1
        d = json.loads(summary.read_text())
        print(json.dumps(d["counts"], indent=2))
        for r in d["results"]:
            print(line(r))
        return 0

    names = sorted(p.stem for p in args.in_dir.glob("*.json"))
    if args.only:
        wanted = set(args.only)
        missing = wanted - set(names)
        if missing:
            print(f"not in {args.in_dir}: {sorted(missing)}")
            return 1
        names = [n for n in names if n in wanted]
    if not names:
        print(f"no json files in {args.in_dir}")
        return 1

    records: dict = {}
    if summary.exists() and not args.force:
        try:
            for r in json.loads(summary.read_text()).get("results", []):
                records[r["name"]] = r
        except (OSError, ValueError):
            pass

    todo = names if args.force else [
        n for n in names
        if not (args.out_dir / f"{n}.json").exists()
        or records.get(n, {}).get("status") in (None, "error", "timeout")]

    jobs = args.jobs or max(1, (os.cpu_count() or 2) // 2)
    jobs = min(jobs, len(todo)) or 1

    print(f"[opt] {len(names)} surfaces in {args.in_dir}")
    print(f"[opt] {len(todo)} to run, {len(names) - len(todo)} already done")
    print(f"[opt] grid {args.samples}x{args.samples}, max_iter {args.max_iter}, "
          f"H_thresh {args.h_thresh}, w_smooth {args.w_smooth:g}, "
          f"H_max {'off' if args.no_h_max else f'p{args.h_max_percentile:g}'}, "
          f"jacobian {'FD' if args.no_jax else 'jax'}")
    if not args.no_skip_satisfied:
        print("[opt] surfaces already meeting the condition are normalised and "
              "copied, not optimised")
    print(f"[opt] {jobs} worker processes, 1 BLAS thread each, "
          f"{args.timeout}s limit per surface")
    print(f"[opt] writing to {args.out_dir}")
    if args.dry_run:
        for n in todo:
            print(f"  would run {n}")
        return 0
    if not todo:
        print("[opt] nothing to do")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(in_dir=str(args.in_dir), out_dir=str(args.out_dir),
                  samples=args.samples, max_iter=args.max_iter,
                  H_thresh=args.h_thresh,
                  h_max_percentile=None if args.no_h_max else args.h_max_percentile,
                  w_max=args.w_max, w_H=args.w_h, w_step=args.w_step,
                  w_smooth=args.w_smooth, factor=args.factor,
                  timeout=args.timeout, use_jax=not args.no_jax,
                  accept_tol=args.accept_tol,
                  skip_satisfied=not args.no_skip_satisfied)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_surfaces": len(names), "settings": common}

    t0 = time.time()
    done = 0
    # spawn, not fork: hanan pulls in jax, and forking an initialised jax is a
    # known way to get a hung child.
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            futures = {pool.submit(run_one, dict(common, name=n)): n for n in todo}
            for fut in as_completed(futures):
                n = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:          # worker died outright
                    rec = {"name": n, "status": "error",
                           "error": f"worker died: {type(e).__name__}: {e}",
                           "seconds": 0.0}
                records[n] = rec
                done += 1
                print(f"[{done}/{len(todo)}]{line(rec)}", flush=True)
                write_summary(summary, meta, records)
    except KeyboardInterrupt:
        print("\n[opt] interrupted; finished surfaces are saved", flush=True)
        write_summary(summary, meta, records)
        return 130

    c = json.loads(summary.read_text())["counts"]
    print(f"\n[opt] {time.time() - t0:.0f}s total")
    print(f"[opt] ok {c['ok']}   already ok {c['already_ok']}   "
          f"incomplete {c['incomplete']}   timeout {c['timeout']}   "
          f"error {c['error']}   of {c['done']} run")
    print(f"[opt] surfaces in {args.out_dir}, summary in {summary}")
    print(f"[opt] 'INC' means it ran but |H| still falls short of "
          f"{args.h_thresh} by more than {args.accept_tol:g} relative, or H "
          f"still changes sign; the 'x thr' column is the margin. see --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
