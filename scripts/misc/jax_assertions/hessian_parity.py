"""
Jax Assertions: LensCalc Hessian np vs jnp Parity
=================================================

Verifies that ``LensCalc.hessian_from`` and
``LensCalc.magnification_2d_via_hessian_from`` produce numerically identical
results on the NumPy (finite-difference) and JAX (``jax.jacfwd``) paths.

The NumPy path uses Richardson extrapolation; the JAX path uses exact
analytic derivatives via ``jax.jacfwd``. They should agree to float64
precision when both can converge.

Previously: two ``test__*_np_jnp_agree_to_float64`` tests in
``test_autogalaxy/operate/test_deflections.py``.
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import autogalaxy as ag
from autogalaxy.operate.lens_calc import LensCalc

"""
__hessian_from: NumPy Richardson Matches JAX jacfwd to float64__
"""
grid = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, 1.0), (0.7, -0.3)])

mp = ag.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.05, -0.111111), einstein_radius=1.5
)

od = LensCalc.from_mass_obj(mp)

np_hess = od.hessian_from(grid=grid, xp=np)
jnp_hess = od.hessian_from(grid=grid, xp=jnp)

for np_component, jnp_component in zip(np_hess, jnp_hess):
    npt.assert_allclose(
        np.asarray(np_component),
        np.asarray(jnp_component),
        rtol=1.0e-8,
    )

"""
__magnification_2d_via_hessian_from: NumPy and JAX Agree to float64__
"""
grid = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, 1.0), (0.7, -0.3)])

mp = ag.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.05, -0.111111), einstein_radius=1.5
)

od = LensCalc.from_mass_obj(mp)

np_mag = od.magnification_2d_via_hessian_from(grid=grid, xp=np)
jnp_mag = od.magnification_2d_via_hessian_from(grid=grid, xp=jnp)

npt.assert_allclose(
    np.asarray(np_mag.array),
    np.asarray(jnp_mag),
    rtol=1.0e-7,
)

print("hessian_parity: all assertions passed")
