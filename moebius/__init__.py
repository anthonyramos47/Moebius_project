"""Moebius project — sphere-congruence & torsal-direction quad mesh optimisation."""

# jax defaults to float32, and nothing in hanan changes that, so every
# jax-differentiated energy term was handing float32 Jacobian entries to a
# float64 Levenberg-Marquardt solve. Second derivatives of a B-spline involve
# heavy cancellation, and in float32 the mean curvature of the reference
# surface came out 2.4e-4 wrong in relative terms, against 1e-15 once this is
# set. It has to be set before the first jax array is created, which is why it
# lives in the package __init__ rather than in the term that needs it.
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

from .pipeline import MoebiusState, load_surface, sample_selection, compute_init
from .pipeline import setup_bspline_optimizer, run_bspline_optimizer
from .pipeline import setup_lc_optimizer, run_lc_optimizer
from .pipeline import setup_torsal_optimizer, run_torsal_optimizer, recompute_torsal_directions
from .pipeline import (
    resolve_remesher_bin, check_remesher, remesher_help,
    torsal_field_per_quad, export_frame_field, save_remesh_input,
    remesher_command, run_remesher,
    load_remeshed_surface, remesh_quality, sweep_gradient,
    fair_remeshed_surface,
)
from .pipeline import setup_postopt_optimizer, run_postopt_optimizer
from .pipeline import save_results, load_state
