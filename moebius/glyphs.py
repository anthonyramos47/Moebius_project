"""
Spherical-panel geometry utilities.

A spherical panel is a geodesic patch on a sphere (c, r) whose boundary is
the intersection of the sphere with a set of tangent planes (one per edge).
For visualisation we approximate the panel as a triangle fan from the sphere
centre projected onto the panel boundary curve.

The main entry point is:

    V, F = spherical_panels_from_mesh(v_mesh, f_mesh, centers, radii)

which returns a single (V, F) triangle mesh with one panel per quad face.

Optionally, the function can be used via kayviz:

    import kayviz as kv
    kv.register_surface_mesh("panels", *spherical_panels_from_mesh(...))
"""

from __future__ import annotations
import numpy as np
import triangle as tr

from hanan.geometry.primitives import project_to_sphere, sphere_inversion
from moebius.utils.spheres import center_radius_from_implicit


# ── spherical panels ─────────────────────────────────────────────────────────
#
# A panel is the piece of a sphere bounded by a closed curve. Triangulating it
# directly on the sphere (fanning geodesic arcs from a corner) gives badly
# shaped triangles and cannot refine the interior.
#
# The L-mesh construction instead flattens the problem exactly. Inversion in a
# sphere centred on the far side of the panel maps the face sphere to a PLANE,
# so the boundary curve becomes a plane curve; that is triangulated in 2D with
# a quality-constrained constrained-Delaunay mesh, and the result is mapped
# back through the same inversion and re-projected onto the sphere. Inversion
# is conformal, so angles — and therefore triangle quality — survive the round
# trip. Ported from Laguerre_project hananLab geo_processing.spherical_panel_no_remesh.

AREA_TOL      = 1e-12   # bounding-box area below which the panel is a point
POLY_AREA_TOL = 1e-12   # true (shoelace) polygon area
ASPECT_TOL    = 1e4     # bbox long side / short side


def _project_to_2d(points_3d):
    """Best-fitting plane of a planar point set, as 2D coordinates in it."""
    centroid = points_3d.mean(axis=0)
    centered = points_3d - centroid
    _, _, Vt = np.linalg.svd(centered)
    u, v = Vt[0], Vt[1]
    return np.column_stack([centered @ u, centered @ v]), centroid, u, v


def _reconstruct_3d(pts_2d, centroid, u, v):
    return centroid + pts_2d[:, 0:1] * u + pts_2d[:, 1:2] * v


def spherical_panel_from_loop(boundary: np.ndarray,
                              center: np.ndarray,
                              radius: float,
                              min_angle: float = 28.0,
                              target_triangles: int = 200):
    """Triangulate the spherical patch bounded by `boundary` on sphere (c, r).

    Returns (V, F); F is None when the panel degenerates to a point.

    The boundary need not lie exactly on the sphere — it is projected first —
    and may have any number of corners, so this works for hex faces as well as
    quads.
    """
    center   = np.asarray(center, dtype=float)
    boundary = project_to_sphere(np.asarray(boundary, dtype=float), center, radius)

    # Invert in a sphere centred beyond the panel's far side: the face sphere
    # passes through that centre, so its image is a plane.
    barycenter = project_to_sphere(
        boundary.mean(axis=0, keepdims=True), center, radius)[0]
    inv_center = barycenter - 2.0 * (barycenter - center)
    inv_radius = 0.8 * radius

    plane_boundary = sphere_inversion(boundary, inv_center, inv_radius)
    pts_2d, centroid, u, v = _project_to_2d(plane_boundary)

    def back(p2d):
        return project_to_sphere(
            sphere_inversion(_reconstruct_3d(p2d, centroid, u, v),
                             inv_center, inv_radius),
            center, radius)

    # Near-duplicate corners make the boundary segments degenerate
    keep = np.ones(len(pts_2d), dtype=bool)
    for i in range(1, len(pts_2d)):
        if np.linalg.norm(pts_2d[i] - pts_2d[i - 1]) < 1e-10:
            keep[i] = False
    pts_2d = pts_2d[keep]

    n = len(pts_2d)
    if n < 3:
        return back(np.zeros((1, 2))), None

    bbox = pts_2d.max(axis=0) - pts_2d.min(axis=0)
    area = bbox[0] * bbox[1]
    if area < AREA_TOL:
        # Triangle reads an underflowed max_area as "no bound" and refines
        # forever, so a degenerate panel must be caught before it is called.
        return back(pts_2d[:1]), None

    # A sliver passes the bbox test but cannot be filled with well-shaped
    # triangles, so quality refinement never terminates. Fan it instead: no
    # Steiner points, always terminates, and at this thinness a fan and a
    # "proper" mesh look the same.
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    poly_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    short, long_ = min(bbox), max(bbox)
    aspect = (long_ / short) if short > 0 else np.inf
    if poly_area < POLY_AREA_TOL or aspect > ASPECT_TOL:
        fan = np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int32)
        return back(pts_2d), fan

    out = _cdt(pts_2d, area / target_triangles, min_angle)
    if out is None:
        return back(pts_2d), None
    return back(out["vertices"]), out["triangles"].astype(np.int32)


def _cdt(pts_2d, max_area, min_angle):
    """Quality-constrained constrained-Delaunay mesh of a closed 2D loop.

    The loop is normalised to unit size before Triangle sees it, and the area
    bound scaled with it. Triangle parses its options out of a command string
    and does NOT understand exponent notation: an option built with %g on a
    small panel becomes e.g. "a1.16e-06", which it reads as a maximum area of
    1.16 — thousands of times the whole panel — so the constraint is silently
    ignored and the "refined" mesh comes back as the 2 triangles of a bare
    quad, with no error raised. Normalising keeps the bound O(1); the fixed
    format is a second guard for very fine targets.
    """
    n = len(pts_2d)
    org = pts_2d.min(axis=0)
    scale = float((pts_2d.max(axis=0) - org).max())
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    pts = (pts_2d - org) / scale
    area = max_area / (scale * scale)

    segments = np.array([[i, (i + 1) % n] for i in range(n)])
    spec = {"vertices": pts, "segments": segments}

    def unscale(out):
        out["vertices"] = out["vertices"] * scale + org
        return out

    try:
        return unscale(tr.triangulate(spec, f"q{min_angle:g}a{area:.12f}Dp"))
    except RuntimeError:
        try:
            return unscale(tr.triangulate(spec, "pD"))  # plain constrained Delaunay
        except RuntimeError:
            return None


def planar_panel_from_loop(boundary: np.ndarray,
                           normal: np.ndarray,
                           offset: float,
                           min_angle: float = 28.0,
                           target_triangles: int = 200):
    """Flat panel for a face whose sphere has degenerated to a plane.

    With A = 0 the normalisation <B,B> - 4AC = 1 gives <B,B> = 1, so B is the
    exact unit normal and C the offset: the plane is <B, x> = C. The boundary
    is projected onto it and triangulated with the same quality CDT used for
    the spherical panels, so a flat panel is meshed as finely as a curved one
    rather than being fanned or dropped.
    """
    n_hat = np.asarray(normal, dtype=float)
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
    b = np.asarray(boundary, dtype=float)
    b = b - (b @ n_hat - offset)[:, None] * n_hat        # exact projection

    pts_2d, centroid, u, v = _project_to_2d(b)
    keep = np.ones(len(pts_2d), dtype=bool)
    for i in range(1, len(pts_2d)):
        if np.linalg.norm(pts_2d[i] - pts_2d[i - 1]) < 1e-10:
            keep[i] = False
    pts_2d = pts_2d[keep]
    if len(pts_2d) < 3:
        return _reconstruct_3d(pts_2d[:1], centroid, u, v), None

    bbox = pts_2d.max(axis=0) - pts_2d.min(axis=0)
    area = bbox[0] * bbox[1]
    if area < AREA_TOL:
        return _reconstruct_3d(pts_2d[:1], centroid, u, v), None

    out = _cdt(pts_2d, area / target_triangles, min_angle)
    if out is None:
        fan = np.array([[0, i, i + 1] for i in range(1, len(pts_2d) - 1)],
                       dtype=np.int32)
        return _reconstruct_3d(pts_2d, centroid, u, v), fan
    return (_reconstruct_3d(out["vertices"], centroid, u, v),
            out["triangles"].astype(np.int32))


def panels_from_spheres(
    v_mesh: np.ndarray,
    f_mesh,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    min_angle: float = 28.0,
    target_triangles: int = 200,
):
    """One panel per face from the implicit sphere coefficients (A, B, C).

    Curved faces get a spherical panel; faces whose sphere has degenerated to a
    plane (|A| ~ 0) get a FLAT panel in that exact plane rather than being
    skipped — a plane is a legitimate panel, and the cheapest one to fabricate.

    Takes (A, B, C) rather than centres and radii so the plane case is visible
    at all: converting first collapses it, since c = B/(2A) and r = 1/(2A) both
    run off to infinity and come back as very large finite numbers.

    `f_mesh` may be an (nf, k) array or a ragged list, so hex and mixed meshes
    work as well as quads. Returns (V, F) with global offsets applied.
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float).reshape(len(A), 3)
    C = np.atleast_1d(np.asarray(C, dtype=float))
    centres, radii, is_plane = center_radius_from_implicit(A, B, C)

    all_V, all_F = [], []
    offset = 0
    n_plane = n_skipped = 0
    for fi, face in enumerate(f_mesh):
        face = np.asarray(face, dtype=np.int32)
        loop = v_mesh[face]

        if is_plane[fi]:
            # <B, x> = C, with B already unit because <B,B> - 4AC = 1 and A = 0
            Vi, Fi = planar_panel_from_loop(loop, B[fi], C[fi],
                                            min_angle=min_angle,
                                            target_triangles=target_triangles)
            n_plane += 1
        else:
            # |r|: the sphere is the same set for either sign, and
            # project_to_sphere would send the loop to the antipode for r < 0.
            r = abs(float(radii[fi]))
            if not np.isfinite(r) or r <= 1e-12:
                n_skipped += 1
                continue
            Vi, Fi = spherical_panel_from_loop(loop, centres[fi], r,
                                                min_angle=min_angle,
                                                target_triangles=target_triangles)
        if Fi is None:
            n_skipped += 1
            continue
        all_V.append(Vi)
        all_F.append(Fi + offset)
        offset += len(Vi)

    if not all_V:
        raise RuntimeError("no panel could be built: every face degenerated")
    if n_plane or n_skipped:
        print(f"  panels: {n_plane} flat, {n_skipped} degenerate face(s) skipped")
    return np.concatenate(all_V, axis=0), np.concatenate(all_F, axis=0)


def spherical_panels_from_mesh(
    v_mesh: np.ndarray,
    f_mesh,
    centers: np.ndarray,
    radii: np.ndarray,
    min_angle: float = 28.0,
    target_triangles: int = 200,
):
    """One spherical panel per face, from centres and radii.

    Kept for callers that only have (centre, radius). Prefer
    `panels_from_spheres`, which takes (A, B, C) and can therefore tell a plane
    from a very large sphere.
    """
    all_V, all_F = [], []
    offset = 0
    skipped = 0
    for fi, face in enumerate(f_mesh):
        face = np.asarray(face, dtype=np.int32)
        r = abs(float(radii[fi]))
        if not np.isfinite(r) or r <= 1e-12:
            skipped += 1
            continue
        Vi, Fi = spherical_panel_from_loop(v_mesh[face], centers[fi], r,
                                            min_angle=min_angle,
                                            target_triangles=target_triangles)
        if Fi is None:
            skipped += 1
            continue
        all_V.append(Vi)
        all_F.append(Fi + offset)
        offset += len(Vi)
    if not all_V:
        raise RuntimeError("no panel could be built: every face degenerated")
    if skipped:
        print(f"  spherical panels: skipped {skipped} degenerate face(s)")
    return np.concatenate(all_V, axis=0), np.concatenate(all_F, axis=0)


def sphere_centers_radii_from_params(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Convert algebraic sphere params (A, B, C) to (centers, radii).

    The sphere equation is  A‖v‖² - B·v + C = 0  (A ≠ 0).
    center = B / (2A),   radius = √(‖B‖²/(4A²) - C/A).
    """
    centers = B.reshape(-1, 3) / (2 * A[:, None])
    r2 = np.einsum("ij,ij->i", centers, centers) - C / A
    radii = np.sqrt(np.maximum(r2, 0))
    return centers, radii
