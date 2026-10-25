"""Focused mass-profile backend regression; run from this workspace root."""
import autogalaxy as ag
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

grid = ag.Grid2DIrregular([(0.1, 0.2), (-0.3, 0.4)])
# A literal singular model is rejected even when constructed while tracing.
def singular(amplitude):
    return ag.mp.PowerLawMultipole(m=1, slope=2., multipole_comps=(amplitude, 0.)).deflections_yx_2d_from(grid, xp=jnp).array
for evaluate in (singular, jax.jit(singular)):
    try:
        evaluate(.01)
    except ag.exc.ModelParameterException as error:
        assert "m=1 / slope=2" in str(error)
    else:
        raise AssertionError("singular dipole was accepted")
# Traced free slopes remain supported away from the documented singular point.
def supported(slope):
    return ag.mp.PowerLawMultipole(m=1, slope=slope, multipole_comps=(.01, 0.)).deflections_yx_2d_from(grid, xp=jnp).array
for slope in (1.9, 2.1):
    expected = ag.mp.PowerLawMultipole(m=1, slope=slope, multipole_comps=(.01, 0.)).deflections_yx_2d_from(grid).array
    np.testing.assert_allclose(jax.jit(supported)(slope), expected, rtol=1e-12)
    assert np.isfinite(jax.jacrev(supported)(slope)).all()
print("PASS: singular literal dipole rejected in eager/JIT; allowed traced slopes preserve parity and gradients")
