#!/usr/bin/env python
"""
Render every B-spline coloured by the SIGN of its mean curvature, as one
contact-sheet PNG and one self-contained HTML page.

    python scripts/curvature_sheet.py                     # data/bsplines
    python scripts/curvature_sheet.py --compare            # input vs optimised
    python scripts/curvature_sheet.py --dir data/bsplines_optimized

The colour scale is pinned symmetrically at zero, so it carries the sign of H
and nothing else: blue where H < 0, red where H > 0. That is the question the
pipeline actually asks of a surface -- r = 1/H blows up where H crosses zero --
and a magnitude scale hides it, because one sharp region saturates the range and
flattens everything else to the midpoint. A surface that is entirely one colour
is usable; any surface showing both colours has a sign change.

Two colours by default, which is the whole point: pure sign. --zero-band draws
a third colour where |H| is small, which is where r = 1/H is largest, but it is
off by default because it is measured against the MEDIAN |H| of the surface, not
the max. Against the max it is useless on exactly the surfaces that matter: on
CM_Strip, whose |H| spikes to 1120, 2% of the max is 22, so the entire rest of
the surface falls in the band and the plot goes solid yellow.

Outputs go to out/curvature/ by default.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moebius.utils.bsplines import (read_bspline_json, sample_bspline_surface,
                                    bspline_curvatures)

NEG, ZERO, POS = "#2c6fbb", "#f0e442", "#c4342b"      # blue / yellow / red


def load(path: Path, samples: int):
    """(X, Y, Z, H) on a samples x samples grid, or None if it cannot evaluate."""
    bsp = read_bspline_json(str(path))
    u = np.linspace(0, 1, samples)
    V, _ = sample_bspline_surface(bsp, u, u)
    _, H, _ = bspline_curvatures(bsp, u, u)
    P = np.asarray(V).reshape(samples, samples, 3)
    return P[..., 0], P[..., 1], P[..., 2], np.asarray(H)


def draw(ax, X, Y, Z, H, zero_band: float, shade: bool = True):
    """Surface coloured by sign(H) only."""
    a = np.abs(H)
    # median, not max: one spike must not define "near zero" for the whole surface
    med = float(np.median(a))
    eps = zero_band * med if (zero_band > 0 and med > 0) else 0.0
    if eps > 0:
        cmap = ListedColormap([NEG, ZERO, POS])
        norm = BoundaryNorm([-np.inf, -eps, eps, np.inf], 3)
    else:
        cmap = ListedColormap([NEG, POS])
        norm = BoundaryNorm([-np.inf, 0.0, np.inf], 2)

    # facecolors are per quad, so take the value at each quad's lower corner
    fc = cmap(norm(H[:-1, :-1]))
    # shade=True keeps the hue (so the colour still reads as the sign) but adds
    # the lighting that makes the form legible; without it every surface is a
    # flat silhouette and you cannot tell what you are looking at.
    ax.plot_surface(X, Y, Z, facecolors=fc, shade=shade,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)

    # equal aspect, so shapes are comparable between surfaces
    c = np.array([X.mean(), Y.mean(), Z.mean()])
    r = max(np.ptp(X), np.ptp(Y), np.ptp(Z)) / 2 or 1.0
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_axis_off()
    ax.view_init(elev=28, azim=-55)


def stats(H):
    h = H.ravel(); a = np.abs(h)
    return dict(hmin=float(h.min()), hmax=float(h.max()), amin=float(a.min()),
                flip=bool(h.min() * h.max() < 0),
                frac_neg=float((h < 0).mean()))


def png_bytes(X, Y, Z, H, zero_band, shade=True, size=(4.2, 3.6), dpi=100):
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, X, Y, Z, H, zero_band, shade)
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=False, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "bsplines")
    ap.add_argument("--compare", action="store_true",
                    help="also show data/bsplines_optimized beside each input")
    ap.add_argument("--compare-dir", type=Path,
                    default=ROOT / "data" / "bsplines_optimized")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--zero-band", type=float, default=0.0,
                    help="draw |H| below this multiple of the surface's MEDIAN "
                         "|H| in a third colour (default 0, off)")
    ap.add_argument("--no-shade", action="store_true",
                    help="flat colour, no lighting")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "curvature")
    ap.add_argument("--cols", type=int, default=9)
    args = ap.parse_args()

    names = sorted(p.stem for p in args.dir.glob("*.json"))
    if not names:
        print(f"no json files in {args.dir}")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for i, n in enumerate(names):
        try:
            d = load(args.dir / f"{n}.json", args.samples)
        except Exception as e:
            skipped.append((n, f"{type(e).__name__}: {e}"))
            continue
        rec = {"name": n, "index": i, "data": d, "stats": stats(d[3]), "opt": None}
        if args.compare:
            q = args.compare_dir / f"{n}.json"
            if q.exists():
                try:
                    d2 = load(q, args.samples)
                    rec["opt"] = {"data": d2, "stats": stats(d2[3])}
                except Exception:
                    pass
        rows.append(rec)
        print(f"  [{i:3d}] {n:24s} H [{rec['stats']['hmin']:+9.3f}, "
              f"{rec['stats']['hmax']:+9.3f}]  "
              f"{'SIGN CHANGE' if rec['stats']['flip'] else 'one-signed'}", flush=True)

    flips = sum(r["stats"]["flip"] for r in rows)
    print(f"\n{len(rows)} rendered, {len(skipped)} skipped; "
          f"{flips} change sign, {len(rows) - flips} one-signed")
    for n, why in skipped:
        print(f"  skipped {n}: {why}")

    # ── contact sheet ────────────────────────────────────────────────────────
    cols = args.cols
    nrow = (len(rows) + cols - 1) // cols
    fig = plt.figure(figsize=(2.6 * cols, 2.5 * nrow), dpi=80)
    for k, r in enumerate(rows):
        ax = fig.add_subplot(nrow, cols, k + 1, projection="3d")
        draw(ax, *r["data"], args.zero_band, not args.no_shade)
        s = r["stats"]
        ax.set_title(f"[{r['index']}] {r['name']}\n"
                     f"{'SIGN CHANGE' if s['flip'] else 'one-signed'}",
                     fontsize=6.5, color=("#b00" if s["flip"] else "#060"), pad=1)
    band = ", yellow |H| near 0" if args.zero_band > 0 else ""
    fig.suptitle(f"Mean curvature sign  —  {args.dir.name}  —  "
                 f"blue H<0, red H>0{band}  "
                 f"({flips} of {len(rows)} change sign)", fontsize=12, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    sheet = args.out_dir / f"sheet_{args.dir.name}.png"
    fig.savefig(sheet, facecolor="white")
    plt.close(fig)
    print(f"\ncontact sheet: {sheet}")

    # ── html ─────────────────────────────────────────────────────────────────
    cards = []
    for r in rows:
        s = r["stats"]
        b64 = base64.b64encode(png_bytes(*r["data"], args.zero_band, not args.no_shade)).decode()
        imgs = f'<img src="data:image/png;base64,{b64}" alt="{html.escape(r["name"])}">'
        extra = ""
        if r["opt"]:
            s2 = r["opt"]["stats"]
            b2 = base64.b64encode(png_bytes(*r["opt"]["data"], args.zero_band, not args.no_shade)).decode()
            imgs += f'<img src="data:image/png;base64,{b2}" alt="optimised">'
            extra = (f'<div class="row"><span>optimised</span>'
                     f'<b class="{"bad" if s2["flip"] else "good"}">'
                     f'{"sign change" if s2["flip"] else "one-signed"}</b></div>'
                     f'<div class="row"><span>H</span><code>[{s2["hmin"]:+.3f}, '
                     f'{s2["hmax"]:+.3f}]</code></div>')
        cards.append(f"""<article class="card" data-flip="{int(s['flip'])}"
     data-name="{html.escape(r['name']).lower()}">
  <div class="imgs">{imgs}</div>
  <h2><span class="idx">{r['index']}</span> {html.escape(r['name'])}</h2>
  <div class="row"><span>input</span><b class="{'bad' if s['flip'] else 'good'}">
     {'sign change' if s['flip'] else 'one-signed'}</b></div>
  <div class="row"><span>H</span><code>[{s['hmin']:+.3f}, {s['hmax']:+.3f}]</code></div>
  <div class="row"><span>min |H|</span><code>{s['amin']:.3e}</code></div>
  <div class="row"><span>H &lt; 0 on</span><code>{100*s['frac_neg']:.1f}% of samples</code></div>
  {extra}
</article>""")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mean Curvature Sign</title>
<style>
  :root {{
    --bg:#fbfbfc; --fg:#1b1d21; --muted:#6b7280; --line:#e4e6ea; --card:#fff;
    --neg:{NEG}; --zero:{ZERO}; --pos:{POS}; --good:#11703a; --bad:#b0221c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#14161a; --fg:#eceef2; --muted:#9aa3b0; --line:#2a2e36; --card:#1b1e24;
      --good:#4ec27f; --bad:#ff7b70;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#14161a; --fg:#eceef2; --muted:#9aa3b0; --line:#2a2e36; --card:#1b1e24;
    --good:#4ec27f; --bad:#ff7b70;
  }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:28px 16px 16px; border-bottom:1px solid var(--line);
    position:sticky; top:0; background:var(--bg); z-index:5 }}
  .wrap {{ max-width:1500px; margin:0 auto; padding:0 16px }}
  h1 {{ margin:0 0 6px; font-size:22px; letter-spacing:-.01em }}
  p.lede {{ margin:0 0 14px; color:var(--muted); max-width:70ch }}
  .key {{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin-bottom:12px }}
  .key b {{ display:inline-flex; align-items:center; gap:6px; font-weight:500; font-size:13px }}
  .sw {{ width:14px; height:14px; border-radius:3px; border:1px solid rgba(0,0,0,.2) }}
  .tools {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center }}
  input, button {{ font:inherit; padding:7px 11px; border:1px solid var(--line);
    border-radius:8px; background:var(--card); color:var(--fg) }}
  button {{ cursor:pointer }}
  button[aria-pressed="true"] {{ background:var(--fg); color:var(--bg); border-color:var(--fg) }}
  .grid {{ display:grid; gap:14px; padding:18px 0 48px;
    grid-template-columns:repeat(auto-fill,minmax(270px,1fr)) }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:10px 12px 12px; }}
  .card[hidden] {{ display:none }}
  .imgs {{ display:flex; gap:4px }}
  .imgs img {{ width:100%; min-width:0; border-radius:8px; background:#fff }}
  h2 {{ font-size:14px; margin:9px 0 7px; display:flex; gap:7px; align-items:baseline;
    word-break:break-word }}
  .idx {{ color:var(--muted); font-variant-numeric:tabular-nums; font-size:12px }}
  .row {{ display:flex; justify-content:space-between; gap:8px; font-size:12.5px;
    padding:1.5px 0; color:var(--muted) }}
  .row code {{ color:var(--fg); font-size:12px }}
  .good {{ color:var(--good) }} .bad {{ color:var(--bad) }}
  .count {{ color:var(--muted); font-size:13px }}
  @media (max-width:560px) {{ .grid {{ grid-template-columns:1fr }} }}
</style></head><body>
<header><div class="wrap">
  <h1>Mean curvature sign &mdash; {html.escape(args.dir.name)}</h1>
  <p class="lede">Colour carries the <strong>sign of H</strong> and nothing else, so a
  single sharp region cannot saturate the scale and flatten the rest. The pipeline
  needs H never to vanish or change sign, because the sphere radii are r = 1/H, so
  a surface in one colour is usable and any surface showing both is not.</p>
  <div class="key">
    <b><i class="sw" style="background:{NEG}"></i>H &lt; 0</b>
    {f'<b><i class="sw" style="background:{ZERO}"></i>|H| below {args.zero_band:g}x the median &mdash; r = 1/H largest here</b>' if args.zero_band > 0 else ''}
    <b><i class="sw" style="background:{POS}"></i>H &gt; 0</b>
  </div>
  <div class="tools">
    <input id="q" type="search" placeholder="filter by name&hellip;" autocomplete="off">
    <button id="f-all"  aria-pressed="true">all</button>
    <button id="f-flip" aria-pressed="false">sign change ({flips})</button>
    <button id="f-ok"   aria-pressed="false">one-signed ({len(rows) - flips})</button>
    <span class="count" id="count"></span>
  </div>
</div></header>
<main class="wrap"><div class="grid" id="grid">
{''.join(cards)}
</div></main>
<script>
  const cards = [...document.querySelectorAll('.card')];
  const q = document.getElementById('q'), count = document.getElementById('count');
  const btns = {{all:document.getElementById('f-all'),
                flip:document.getElementById('f-flip'),
                ok:document.getElementById('f-ok')}};
  let mode = 'all';
  function apply() {{
    const t = q.value.trim().toLowerCase();
    let shown = 0;
    for (const c of cards) {{
      const flip = c.dataset.flip === '1';
      const okMode = mode === 'all' || (mode === 'flip') === flip;
      const okText = !t || c.dataset.name.includes(t);
      const vis = okMode && okText;
      c.hidden = !vis;
      if (vis) shown++;
    }}
    count.textContent = shown + ' of ' + cards.length + ' shown';
    for (const k in btns) btns[k].setAttribute('aria-pressed', String(k === mode));
  }}
  for (const k in btns) btns[k].addEventListener('click', () => {{ mode = k; apply(); }});
  q.addEventListener('input', apply);
  apply();
</script>
</body></html>"""
    page = args.out_dir / f"curvature_{args.dir.name}.html"
    page.write_text(doc)
    print(f"html page    : {page}  ({len(doc)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
