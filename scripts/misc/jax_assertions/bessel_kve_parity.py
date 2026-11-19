"""
Jax Assertions: Vendored Bessel K Parity (``bessel_kve`` / ``kv``)
=================================================================

The JAX Matérn-kernel regularization evaluates the modified Bessel function of
the second kind ``K_nu(z)`` for arbitrary real order: ``nu`` is a continuous free
prior (``Uniform(0.5, 5.5)``), so closed-form half-integer expressions are not
enough, and ``jax.scipy.special`` has no ``kv``/``kve``. PyAutoArray therefore
vendors a pure-``jax.numpy`` port of tensorflow-probability's ``bessel_kve``
(``autoarray/inversion/regularization/bessel.py``, PyAutoArray#694), replacing
the former ``tfp-nightly`` dependency.

This script is the numerical certification of that port. It checks:

* ``bessel_kve`` and ``kv`` against ``scipy.special.kve`` / ``kv`` to
  ``rtol=1e-10`` on a ``(nu, z)`` grid spanning the prior range, with extra
  points straddling every regime boundary of the algorithm (Temme power series
  for ``z <= 2`` / Steed continued fraction for ``z > 2``; half-integer orders,
  where the forward-recurrence count changes; and ``nu >= 50``, where Olver's
  uniform asymptotic expansion takes over).
* the same under ``jax.jit`` and ``jax.vmap`` (the regularization path is
  traced, not eager).
* ``jax.grad`` of ``kv`` with respect to ``z`` against a central finite
  difference of SciPy's ``kv`` and against the closed form
  ``dK_nu/dz = -(K_{nu-1} + K_{nu+1}) / 2``, including the second derivative.
* that no ``tensorflow_probability`` module is imported by the JAX Matérn path.

Known limitation (matches tfp; documented in ``kv_xp``): no derivative is
defined with respect to the order ``nu``, so ``nu`` must be treated as a static
float when differentiating the Matérn kernel.
"""

import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import scipy.special as sc

from autoarray.inversion.regularization.bessel import bessel_kve, kv
from autoarray.inversion.regularization.matern_kernel import (
    kv_xp,
    matern_cov_matrix_from,
)

"""
__Grid: the Matérn prior range plus every regime boundary__

``nu`` covers ``[0.5, 5.5]`` densely, plus orders just either side of
half-integers (the recurrence count ``n = round(nu)`` changes there) and the
``nu >= 50`` Olver regime. ``z`` spans ``1e-6`` to ``500`` log-uniformly plus
points just either side of the ``z = 2`` series / continued-fraction switch.
"""
nu_values = np.concatenate(
    [
        np.linspace(0.5, 5.5, 51),
        [0.4999, 0.5001, 1.4999, 1.5001, 2.4999, 2.5001, 4.4999, 4.5001],
        [49.9, 50.0, 60.0],
    ]
)
z_values = np.concatenate(
    [np.geomspace(1.0e-6, 500.0, 60), [1.999, 1.9999999, 2.0, 2.0000001, 2.001]]
)
nu_grid, z_grid = np.meshgrid(nu_values, z_values, indexing="ij")

"""
__Values: jitted ``bessel_kve`` / ``kv`` match SciPy to rtol=1e-10__

Points where SciPy itself overflows to ``inf`` (large ``nu`` at tiny ``z``) are
excluded from the comparison: there the vendored result is either ``inf`` too or
a finite value just below the float64 limit (~1e303-1e307), never ``nan``.
Wherever SciPy is finite, the vendored value must be finite and agree.
"""
kve_jax = np.asarray(jax.jit(bessel_kve)(nu_grid, z_grid))
kve_ref = sc.kve(nu_grid, z_grid)

finite = np.isfinite(kve_ref)
assert not np.any(np.isnan(kve_jax)), "vendored kve returned nan"
assert np.all(
    np.isfinite(kve_jax[finite])
), "vendored kve overflows where SciPy does not"
npt.assert_allclose(kve_jax[finite], kve_ref[finite], rtol=1.0e-10)

kv_jax = np.asarray(jax.jit(kv)(nu_grid, z_grid))
kv_ref = sc.kv(nu_grid, z_grid)
finite_kv = np.isfinite(kv_ref) & (kv_ref > 0.0)
npt.assert_allclose(kv_jax[finite_kv], kv_ref[finite_kv], rtol=1.0e-10)

print(
    "bessel_kve parity: max rel err "
    f"{np.max(np.abs(kve_jax[finite] - kve_ref[finite]) / kve_ref[finite]):.2e} "
    f"over {finite.sum()} points"
)

"""
__vmap: scalar evaluation mapped over the grid agrees with the batched call__
"""
kve_vmap = np.asarray(jax.vmap(bessel_kve)(nu_grid.ravel(), z_grid.ravel()))
npt.assert_allclose(
    kve_vmap.reshape(nu_grid.shape)[finite], kve_jax[finite], rtol=1.0e-12
)

"""
__kv_xp: the Matérn entry point dispatches to the vendored function under JAX__
"""
prior_mask = (nu_grid >= 0.5) & (nu_grid <= 5.5) & finite_kv
npt.assert_allclose(
    np.asarray(kv_xp(nu_grid, z_grid, xp=jnp))[prior_mask],
    sc.kv(nu_grid, z_grid)[prior_mask],
    rtol=1.0e-10,
)

"""
__Gradient in z: jax.grad vs central finite difference and the closed form__

The finite difference uses a relative step on SciPy's ``kv`` (accurate to
~1e-7); the closed form ``-(K_{nu-1} + K_{nu+1}) / 2`` gives a tight check.
Restricted to the prior range and to ``z >= 1e-3`` so that ``K_nu`` and its
derivative stay well inside float64 range.
"""
grad_mask = (nu_grid >= 0.5) & (nu_grid <= 5.5) & (z_grid >= 1.0e-3)
nu_g = nu_grid[grad_mask]
z_g = z_grid[grad_mask]

dkv_dz = jax.jit(jax.vmap(jax.grad(kv, argnums=1)))
grad_jax = np.asarray(dkv_dz(jnp.asarray(nu_g), jnp.asarray(z_g)))

step = 1.0e-6 * z_g
grad_fd = (sc.kv(nu_g, z_g + step) - sc.kv(nu_g, z_g - step)) / (2.0 * step)
grad_closed = -(sc.kv(nu_g - 1.0, z_g) + sc.kv(nu_g + 1.0, z_g)) / 2.0

nonzero = np.abs(grad_closed) > 1.0e-250
npt.assert_allclose(grad_jax[nonzero], grad_fd[nonzero], rtol=1.0e-5)
npt.assert_allclose(grad_jax[nonzero], grad_closed[nonzero], rtol=1.0e-10)

"""
__Second derivative in z__

``d^2 K_nu / dz^2 = (K_{nu-2} + 2 K_nu + K_{nu+2}) / 4``; the custom JVP is
recursive, so higher-order autodiff must also be exact.
"""
d2kv_dz2 = jax.jit(jax.vmap(jax.grad(jax.grad(kv, argnums=1), argnums=1)))
hess_jax = np.asarray(d2kv_dz2(jnp.asarray(nu_g), jnp.asarray(z_g)))
hess_closed = (
    sc.kv(nu_g - 2.0, z_g) + 2.0 * sc.kv(nu_g, z_g) + sc.kv(nu_g + 2.0, z_g)
) / 4.0
nonzero_h = np.abs(hess_closed) > 1.0e-250
npt.assert_allclose(hess_jax[nonzero_h], hess_closed[nonzero_h], rtol=1.0e-9)

"""
__End to end: the Matérn covariance builds and differentiates without tfp__

Gradients with respect to ``scale`` flow through ``kv``'s argument; the JAX
covariance matches the NumPy one. No ``tensorflow_probability`` module may be
imported along the way.
"""
rng = np.random.default_rng(0)
pixel_points = rng.standard_normal((12, 2))


def covariance_sum(scale):
    return jnp.sum(
        matern_cov_matrix_from(
            scale=scale, nu=1.7, pixel_points=jnp.asarray(pixel_points), xp=jnp
        )
    )


grad_scale = float(jax.grad(covariance_sum)(0.7))
h = 1.0e-6
grad_scale_fd = float((covariance_sum(0.7 + h) - covariance_sum(0.7 - h)) / (2.0 * h))
npt.assert_allclose(grad_scale, grad_scale_fd, rtol=1.0e-6)

npt.assert_allclose(
    np.asarray(
        matern_cov_matrix_from(
            scale=0.7, nu=1.7, pixel_points=jnp.asarray(pixel_points), xp=jnp
        )
    ),
    matern_cov_matrix_from(scale=0.7, nu=1.7, pixel_points=pixel_points, xp=np),
    rtol=1.0e-10,
)

tfp_modules = [
    name for name in sys.modules if name.startswith("tensorflow_probability")
]
assert not tfp_modules, f"tensorflow_probability imported: {tfp_modules[:3]}"

print("bessel_kve_parity: all assertions passed")
