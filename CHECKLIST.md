# Checklist

Open work on the MoebiusV2 pipeline, as of 2026-10-01.

## Remesher

The pipeline calls the external quad-remesher binary at `state.remesher_bin`
(default: `~/KAUST/QuadRemesher/Quad_Remesher/build/quadRemesher`).
See `../Laguerre_project/CHECKLIST.md` for the full remesher status across versions.

- [ ] Verify the cross-field export format in `save_remesh_input()`: the file
      `crossfield.txt` writes one `t1 t2` pair per quad face (6 floats/line); confirm
      the remesher binary actually reads this format (check against Lconjugacy's export)
- [ ] Test a full remesh run end-to-end: `save_remesh_input` → `run_remesher` →
      `load_remeshed_surface`; check the remeshed OBJ loads as a valid quad mesh
- [ ] Verify LC interpolation after remeshing: `interpolate_lc` uses barycentric
      coordinates on the triangulated sampling mesh; check that the interpolated
      directions are smooth and correctly oriented
- [ ] Check the remesher handles the case of no singularities (torsal field should
      be singularity-free if the optimisation converged well); add a singularity
      count diagnostic before calling the remesher
- [ ] Hard-coded `DEFAULT_REMESHER` path in `pipeline.py` — replace with a
      platform-resolved path or an env variable (`REMESHER_BIN`)

## MoebiusV2 pipeline — remaining checks

- [ ] Run all notebook cells top-to-bottom on at least two surfaces (CMC + one
      with sign changes in H) and verify the output OBJs are watertight quads
- [ ] B-spline optimisation (step 1b): tune `H_THRESH` default; verify the surface
      deformation is visually imperceptible for well-conditioned inputs
- [ ] Post-optimisation: `SphereFit` initial parameters (A=1, B=face centroid, C=0)
      are a rough guess — check convergence and whether a better initialisation
      (from the optimised r(u,v) field) improves it
- [ ] `SupportPlanarity` uses JAX autodiff through a Python loop — profile and
      consider an analytical Jacobian if post-opt is slow
- [ ] `save_results` writes `sphere_panels.obj`; check the panel geometry is
      correct (especially near boundary faces where spheres may be large)

## Chakana_Geo submodule

- [ ] After any push to `Chakana_Geo/`, update the pinned commit here:
      `git -C Chakana_Geo pull && git add Chakana_Geo && git commit -m "Update Chakana_Geo submodule"`
- [ ] New clones need `git clone --recurse-submodules`; document in README
