"""Focused mass-profile backend regression; run from this workspace root."""
import autogalaxy as ag
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from time import perf_counter

grid = ag.Grid2DIrregular([(0., 1.), (1., 0.), (0., -1.), (-1., 0.), (.1, .2)])
for slope in (1.7, 2., 2.3):
    for ell_comps in ((0., 0.), (0., .2), (.1, .2)):
        profile = ag.mp.PowerLaw(ell_comps=ell_comps, slope=slope)
        expected = profile.deflections_yx_2d_from(grid).array
        eager = profile.deflections_yx_2d_from(grid, xp=jnp).array
        compiled = jax.jit(lambda: profile.deflections_yx_2d_from(grid, xp=jnp).array)
        actual = compiled()
        for result in (eager, actual):
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=0.)
            assert np.all(np.asarray(result)[expected == 0.] == 0.)

# A nonzero core regularizer already defines the exact-centre behavior; retain it
# without introducing NaN reverse-mode derivatives through the unit-vector norm.
centre_grid = ag.Grid2DIrregular([(0., 0.), (.1, .2)])
def at_centre(centre):
    return ag.mp.PowerLaw(centre=centre, ell_comps=(0., .2)).deflections_yx_2d_from(centre_grid, xp=jnp).array
assert np.isfinite(jax.jacrev(at_centre)(jnp.zeros(2))).all()
large_grid = ag.Grid2DIrregular(np.random.default_rng(0).uniform(.01, 2., (10000, 2)))
def field(radius):
    return ag.mp.PowerLaw(einstein_radius=radius, ell_comps=(0., .2)).deflections_yx_2d_from(large_grid, xp=jnp).array
compiled_field = jax.jit(field)
compiled_field(1.).block_until_ready()
timings = []
for _ in range(100):
    start = perf_counter()
    compiled_field(1.).block_until_ready()
    timings.append(perf_counter() - start)
print(f"PASS: PowerLaw axial zeros and backend parity; finite centre jacrev; median JIT {np.median(timings)*1e6:.1f} us / 10000 points")
