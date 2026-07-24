"""
Jax Assertions: Matern-Kernel Regularization (tfp bessel_kve path)
==================================================================

Verifies that ``MaternKernel`` regularization builds end to end on the JAX
backend and matches the eager NumPy reference. This is the regression guard for
the 2026-07-13 release-validation failure (PyAutoArray #385): the JAX Matern
path lazily imports ``tensorflow_probability.substrates.jax`` for the
modified-Bessel ``bessel_kve`` term, and the last *stable* tfp release (0.25.0)
crashes at import under the resolved ``jax>=0.7`` stack
(``jax.interpreters.xla.pytype_aval_mappings`` was removed from modern JAX).
The fix pins the tfp nightly; this script fails loudly if that combination ever
regresses.

It exercises:

* ``matern_cov_matrix_from`` + ``inv_via_cholesky`` (the regularization-matrix
  math) with ``xp=np`` vs ``xp=jnp``, agreeing to ``atol=1e-8``.
* the same under ``jax.jit`` (confirms ``kv_xp`` traces, not just runs eagerly).
* ``MaternKernel.regularization_matrix_from`` — the real entry point used by
  pixelizations — via a minimal mesh-grid stub.
* two ``nu`` values: a half-integer (0.5) and a non-half-integer (1.7). ``nu``
  is a continuous free prior (``Uniform(0.5, 5.5)``), so the Bessel term must be
  correct for arbitrary real order, not just closed-form half-integers.

Library unit tests stay numpy-only by project convention; this cross-xp
coverage lives here in workspace_test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import autoarray as aa
from autoarray.inversion.regularization.matern_kernel import (
    matern_cov_matrix_from,
    inv_via_cholesky,
)


"""
__Setup: a small irregular set of source-plane pixel coordinates [N, 2]__
"""
rng = np.random.default_rng(0)
pixel_points = rng.standard_normal((12, 2)).astype(np.float64)

scale = 0.7


"""
__matern_cov_matrix_from + inv_via_cholesky: fp64 JAX matches NumPy to atol=1e-8__

Runs both nu = 0.5 (half-integer) and nu = 1.7 (arbitrary real order) through
the tfp bessel_kve path.
"""
for nu in (0.5, 1.7):
    cov_np = matern_cov_matrix_from(
        scale=scale, nu=nu, pixel_points=pixel_points, xp=np
    )
    reg_np = inv_via_cholesky(cov_np, xp=np)

    cov_jnp = matern_cov_matrix_from(
        scale=scale, nu=nu, pixel_points=jnp.asarray(pixel_points), xp=jnp
    )
    reg_jnp = inv_via_cholesky(cov_jnp, xp=jnp)

    npt.assert_allclose(np.asarray(cov_np), np.asarray(cov_jnp), atol=1.0e-8)
    npt.assert_allclose(np.asarray(reg_np), np.asarray(reg_jnp), atol=1.0e-8)


"""
__Under jax.jit: kv_xp traces (not just runs eagerly)__

A tracer flowing through ``kv_xp`` is what would surface a broken tfp import at
compile time; assert the jitted result still matches NumPy.
"""


def _reg_matrix_jax(points, scale, nu):
    cov = matern_cov_matrix_from(scale=scale, nu=nu, pixel_points=points, xp=jnp)
    return inv_via_cholesky(cov, xp=jnp)


reg_jit = jax.jit(_reg_matrix_jax, static_argnums=(1, 2))(
    jnp.asarray(pixel_points), scale, 1.7
)

reg_np_ref = inv_via_cholesky(
    matern_cov_matrix_from(scale=scale, nu=1.7, pixel_points=pixel_points, xp=np),
    xp=np,
)
npt.assert_allclose(np.asarray(reg_jit), np.asarray(reg_np_ref), atol=1.0e-8)


"""
__MaternKernel.regularization_matrix_from: the real pixelization entry point__

The class reads ``linear_obj.source_plane_mesh_grid.array`` (an [N, 2] coord
array) and ``linear_obj.params``; a minimal stub supplies just those.
"""


class _MeshGridStub:
    def __init__(self, array):
        self.array = array


class _LinearObjStub:
    def __init__(self, points):
        self.source_plane_mesh_grid = _MeshGridStub(points)
        self.params = points.shape[0]


regularization = aa.reg.MaternKernel(coefficient=1.0, scale=scale, nu=1.7)

reg_matrix_np = regularization.regularization_matrix_from(
    linear_obj=_LinearObjStub(pixel_points), xp=np
)
reg_matrix_jnp = regularization.regularization_matrix_from(
    linear_obj=_LinearObjStub(jnp.asarray(pixel_points)), xp=jnp
)

npt.assert_allclose(np.asarray(reg_matrix_np), np.asarray(reg_matrix_jnp), atol=1.0e-8)


print("matern_regularization: all assertions passed")
