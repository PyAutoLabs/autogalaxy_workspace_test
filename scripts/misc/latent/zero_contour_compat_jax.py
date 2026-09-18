"""Zero-contour compatibility regression for JAX 0.10 and 0.11.

Run from the workspace root. Circle paths supply an analytic value and
derivative oracle for the optional solver; LensCalc exercises both production
construction sites. The short, unterminated circle trace checks transformations
including vmap, without promising vmap support for arbitrary terminated paths
(which the upstream solver explicitly excludes).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax_zero_contour import ZeroSolver

import autogalaxy as ag
from autogalaxy.operate._zero_contour import zero_solver_type

jax.config.update("jax_enable_x64", True)

upstream_step = ZeroSolver.step_parallel_tol
solver_type = zero_solver_type()
solver = solver_type(tol=1e-10, max_newton=10)
assert zero_solver_type() is solver_type
assert ZeroSolver.step_parallel_tol is upstream_step
assert solver_type is not ZeroSolver


def residual(radius, point):
    return jnp.sum(point**2) - radius**2


def contour(radius):
    paths, _ = solver.zero_contour_finder(
        jax.tree_util.Partial(residual, radius),
        jnp.array([[1.1, 0.0]]),
        delta=0.1,
        N=8,
    )
    return paths["path"]


def recovered_radius(radius):
    return jnp.mean(jnp.linalg.norm(contour(radius), axis=-1))


radii = jnp.array([0.9, 1.0, 1.1])
for radius in radii:
    eager = contour(radius)
    compiled = jax.jit(contour)(radius)
    assert np.isfinite(compiled).all()
    np.testing.assert_allclose(compiled, eager, atol=1e-9)
    np.testing.assert_allclose(
        jnp.linalg.norm(compiled, axis=-1), radius, atol=1e-9
    )

np.testing.assert_allclose(
    jax.jit(jax.vmap(recovered_radius))(radii), radii, atol=1e-9
)
np.testing.assert_allclose(
    jax.jit(jax.vmap(jax.grad(recovered_radius)))(radii), 1.0, atol=1e-8
)
# Directly exercise a plain partial at the boundary that custom_root changed.
projected, _ = solver.step_parallel_tol(
    partial(residual, 1.0), jnp.array([1.1, 0.0]), factor=2
)
np.testing.assert_allclose(jnp.linalg.norm(projected), 1.0, atol=1e-9)


def einstein_radius(radius):
    calc = ag.LensCalc.from_mass_obj(ag.mp.IsothermalSph(einstein_radius=radius))
    return calc.einstein_radius_jit_from(
        init_guess=jnp.array([[0.0, 1.1]]), delta=0.025, N=400, tol=1e-8
    )


compiled_radius = jax.jit(einstein_radius)
for radius in (1.0, 1.2):
    # Polygon-area estimation omits the final NaN-padded closing edge; this
    # discretization tolerance is independent of the callable compatibility fix.
    np.testing.assert_allclose(compiled_radius(radius), radius, rtol=0.01)

calc = ag.LensCalc.from_mass_obj(ag.mp.IsothermalSph(einstein_radius=1.0))
curves = calc.tangential_critical_curve_list_via_zero_contour_from(
    init_guess=jnp.array([[0.0, 1.1]]), delta=0.025, N=400, tol=1e-8
)
assert len(curves) == 1
points = np.asarray(curves[0])
assert points.shape[0] > 100
np.testing.assert_allclose(np.linalg.norm(points, axis=-1), 1.0, atol=1e-6)
cached_solver = next(iter(calc._zero_contour_cache.values()))[1]
calc.einstein_radius_jit_from(
    init_guess=jnp.array([[0.0, 1.1]]), delta=0.025, N=400, tol=1e-8
)
assert next(iter(calc._zero_contour_cache.values()))[1] is cached_solver
assert ZeroSolver.step_parallel_tol is upstream_step
print(f"PASS: JAX {jax.__version__}: circle values/jit/vmap/grad; both LensCalc paths")
