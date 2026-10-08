# Moebius project — approximation by meshes with spherical faces

Approximates a reference B-spline surface by a quad mesh whose faces can be
realised as **spherical panels** with a **planar support structure**: a sphere
congruence is optimised so that its associated line congruence has real torsal
directions, those directions drive a quad remeshing, and a final optimisation
fits one sphere per face.

This is an **improved implementation** of

> A. S. Ramos Cisneros, A. Aikyn, M. Kilian, H. Pottmann, C. Müller,
> *Approximation by Meshes with Spherical Faces*,
> ACM Transactions on Graphics 43(6), Article 179 (SIGGRAPH Asia 2024).
> [doi:10.1145/3687942](https://doi.org/10.1145/3687942)

Section 5 of the paper is the reference for the pipeline, and Tables 1–3 give
the published weights.

## What is improved over the published implementation

**Inequality constraints use a different formulation.** The paper writes each
one-sided constraint with a dummy variable — `⟨l,n⟩² − cos²θ − μ² = 0` and
`⟨n₁,n₂⟩² − cos²α + ν² = 0`. That has a stationary trap: the derivative with
respect to the slack is `±2μ`, so a slack sitting at zero has a vanishing
Jacobian column and can never leave zero. Measured on real runs, `μ` was pinned
at exactly 0 on over 90% of grid points and `θ` on *every* face, which silently
turned the torsal-angle inequality into the **equality** `angle = α`, dragging
well-separated torsal planes back down to the threshold.

Here the same constraints are **one-sided hinge residuals**, `max(0, ·)`, with
analytic gradients and no slack variables at all. The residual has a kink at the
constraint boundary but the energy `Σ r²` is C¹ there, so Levenberg–Marquardt is
well behaved. On the `Tunel` reference surface this removed 761 unknowns and
~400 singular directions from the system, and took the angle-threshold violation
from 35% of points to 0%.

The dummy-variable form is still available (`hinge=False`) for reproducing the
paper exactly.

**Bad frames are dropped rather than forced on the remesher.** A torsal
direction is only meaningful where the torsal quadratic has a real solution; the
optimisation leaves a handful of faces where it does not, and near the patch
boundary the field can disagree sharply with its neighbours. Feeding such a
frame to the remesher is worse than feeding it nothing — one rogue direction
locally deflects the quads around it. The remesher interpolates the field across
any face absent from its `--indices` list (libigl tutorial 506 style), so
`export_frame_field(..., filter_outliers=True)` simply leaves them out: faces
whose torsal directions were never real, frames disagreeing with their grid
neighbours, near-degenerate frames, and optionally rings along the boundary.
Each criterion is reported with a count, and on the reference surfaces this
drops on the order of 1% of faces.

**Control over the boundary.** The remeshed quad mesh is the thing being
deformed, so its boundary drifts: the patch edge is where the line congruence is
least constrained and where the fairness stencil is one-sided. Variables can now
be held *exactly* fixed — a hard constraint, not a penalty — through
`Optimizer.fix_variables` in the `hanan` library, and the pipeline exposes it as
`fix_boundary` in both the fairing and post-optimisation stages. With it the
boundary curve is preserved bit-for-bit while the interior is free to move.

## Installation

Needs [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html).

```bash
git clone --recurse-submodules <this repo>     # Chakana_Geo must be present
cd Moebius_project
conda env create -f environment.yaml           # creates "Mproj", python 3.11
conda activate Mproj
```

Run `conda env create` **from the repository root**: the spec ends with two
editable installs of the `Chakana_Geo` submodule (`hanan`, the geometry and
optimisation library, and `kayviz`, the viewer), and their relative paths
resolve against the working directory.

Only `python` and `pip` come from conda; everything else is a prebuilt wheel, so
nothing compiles and the environment builds in about ninety seconds.

### The remesher

**A built remesher is included**, at `bin/quadRemesher`, so a fresh clone runs
the whole pipeline with nothing else to compile. It is picked up automatically —
`resolve_remesher_bin()` checks `bin/` first, then a sibling `../QuadRemesher`
checkout, then `$REMESHER_BIN` and `$PATH` — and `check_remesher()` verifies it
understands the current flag-based CLI.

| platform | status |
|---|---|
| **Linux x86-64** | provided, `bin/quadRemesher` |
| **Windows** | to come |
| **macOS** | to come |

The bundled binary is linked only against system libraries (libblas, libm,
libc, libgcc, libstdc++), so it should run on any comparable Linux without
extra setup. Windows and macOS builds will be added later; until then, build
the remesher from source —

> **https://github.com/anthonyramos47/QuadRemesher**

— and point `state.remesher_bin` or `$REMESHER_BIN` at the result. That
repository's README covers the build and the full command-line interface.

## Running

```bash
jupyter lab notebooks/moebius_pipeline.ipynb
```

The notebook is the driver — `moebius/` is deliberately GUI-free and can also be
scripted directly. Each stage begins with a block of named parameters and the
reasoning for their values.

| stage | what it does |
|---|---|
| 0–1 | read a B-spline from `data/bsplines/`, sample it, initialise the central sphere congruence `r = 1/H` |
| 1b | optionally deform the surface so `\|H\|` stays inside `[H_thresh, H_max]`, keeping `r = 1/H` well conditioned without creasing |
| 2 | move the radii so the line congruence is orthogonal to the sphere congruence and within θ of the normal |
| 3 | optimise the torsal directions, recomputing them analytically every *N* steps |
| 4 | export the frame field and remesh along it; outlier frames can be dropped and are interpolated by the remesher |
| 4b | glide the remeshed mesh over the reference surface with fairness, giving the fair quad mesh *Q′* |
| 5 | fit one sphere per face, with support planarity, torsal planes and proximity |
| 6 | write the OBJs — surface, sphere centres, remeshed, optimised, spherical panels — and the state |

Outputs go to `notebooks/out/<surface>/<experiment>/`.

### Which surfaces this works on

**The reference surface must have a mean curvature that never vanishes, and
never changes sign.** The pipeline starts from the central sphere congruence,
whose radii are `r = 1/H`, so an `H` that passes through zero sends the radii
through infinity and there is nothing to optimise. This is a limitation of the
method itself, not of the implementation (paper, Sec. 6.3).

Stage 1b offers an optimisation that deforms the surface's control points to
push `|H|` above a threshold. It obviously helps where `H` merely comes close to
zero, but it can also remove an outright sign change, as long as the region on
the wrong side is small: on `surface_00002`, whose sampled `H` spans
`[-6.84, +6.05]`, it drove the 12 offending samples across and left `H` entirely
negative (`|H|min` 3.7e-03 → 5.0e-02) while moving the surface by a median of
1.1e-06 and a maximum of 9.3e-03. What decides the outcome is how much of the
surface has to move, not whether the sign formally changes — a shape with a
large region of the opposite sign cannot be nudged out of it, and the stage will
either fail to converge or return something no longer resembling the input.
Either way, check `H` before and after.

The stage takes a **ceiling** on the curvature as well as a floor, `H_max`.
Without one, the cheapest way to satisfy the floor at a near-flat point is a
local dent: it raises `|H|` exactly where it is needed and changes nothing
else, and the result reads as a crease. The floor alone has no preference for a
broad deformation over a sharp one, and `set_lap_smooth` — the only other thing
resisting it — damps its own weight to zero after ten iterations, so it does not
constrain the second half of a longer run. The ceiling is a hinge inside the
energy term, so it is active throughout, and capping `|H|` also bounds the
radii from below, keeping the congruence inside `1/H_max <= r <= 1/H_thresh`
instead of letting a crease produce arbitrarily small spheres. It is an absolute
curvature, so the explorer notebook prints the surface's own `|H|` percentiles
and a suggested value; `None` disables it.

`notebooks/bspline_explorer.ipynb` does exactly that: it reports `H`/`K` ranges
and sign changes for one surface or for the whole folder, lists the surfaces
that are directly usable, and runs stage 1b on the rest so a conditioned copy
can be saved back out.

Control points in `data/bsplines/` are stored **normalised** — centred on the
origin with the longest bounding-box dimension equal to 2, via Chakana's
`normalize_vertices`. The loader applies that anyway, so this changes no
results; it means the stored data matches what the pipeline uses, that reading
with `normalize=False` now gives the same surface, and that a curvature
threshold means the same thing from one surface to the next. Section 5 of the
explorer notebook re-applies it, and is idempotent.

`data/bsplines/` holds the surfaces this has been run on — `Tunel.json` is the
default, with `rot`, `tunel_inv`, `tunel_inv_1` and `surface_00002` also
verified end to end. More examples will be added.

The folder still needs a pass: it also carries older B-splines that have not
been checked against the condition above, and some of them certainly violate
it. Only the ones listed as verified are known to have a one-signed `H`. The
intention is to reduce `data/bsplines/` to surfaces with `H != 0` throughout,
so that anything in it can be run without first testing whether the method
applies to it.

### Choosing surfaces by eye

```bash
python scripts/triage_bsplines.py
```

Shows one B-spline at a time with its curvature fields and a **Keep** / **Skip**
pair of buttons in the viewer panel, writing each verdict to
`data/bsplines_triage.json` as it is made, so the session can be interrupted and
resumed. `--report` prints the lists, `--all` revisits decided surfaces, `Back`
undoes. It records the two lists and deletes nothing.

## Layout

- `moebius/` — the pipeline. `pipeline.py` holds the stages, `energies/` one
  `ObjectiveTerm` per file, `utils/` B-spline and sphere helpers, `glyphs.py` the
  spherical panels.
- `Chakana_Geo/` — submodule: `hanan` (meshes, I/O, Levenberg–Marquardt
  optimiser) and `kayviz` (browser-based viewer).
- `notebooks/` — the pipeline notebook.
- `data/bsplines/` — reference surfaces.
- `scripts/triage_bsplines.py` — keep/skip triage of the input surfaces.
- `bin/quadRemesher` — the bundled quad remesher used by stage 4.

*The name Hanan comes from the Kichwa cosmovision, where Hanan-Pacha refers to
the spiritual world.*

---

This clean version of the code was cleaned and organized with the help of AI
(Claude Opus 5.5).
