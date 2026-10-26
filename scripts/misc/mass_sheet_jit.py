"""Focused mass-profile backend regression; run from this workspace root."""

import autogalaxy as ag
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

coordinates = np.array([(0.1, 0.2), (-0.3, 0.4), (0.0, 0.0)])
centre = (0.03, -0.02)
grid = ag.Grid2DIrregular(coordinates)


def deflections(kappa):
    return (
        ag.mp.MassSheet(centre=centre, kappa=kappa)
        .deflections_yx_2d_from(grid, xp=jnp)
        .array
    )


compiled = jax.jit(deflections)
for kappa in (-0.5, 0.0, 0.5, 2.0):
    expected = kappa * (coordinates - np.asarray(centre))
    numpy = (
        ag.mp.MassSheet(centre=centre, kappa=kappa).deflections_yx_2d_from(grid).array
    )
    for result in (numpy, deflections(kappa), compiled(kappa)):
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
np.testing.assert_allclose(jax.jacfwd(compiled)(0.5), coordinates - centre, atol=1e-14)
np.testing.assert_allclose(
    jax.jit(jax.vmap(deflections))(jnp.array([-0.5, 0.5])),
    np.array([-0.5, 0.5])[:, None, None] * (coordinates - centre),
    atol=1e-14,
)
print(
    "PASS: MassSheet NumPy/eager JAX/JIT/vmap and derivative match kappa*(grid-centre)"
)
