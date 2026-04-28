"""
JAX Likelihood: Parametric Light Profile (Interferometer)
=========================================================

Verify that JAX can compute the log-likelihood of an ``Interferometer`` fit for
an autogalaxy model composed of a Sersic bulge. Two paths are exercised:

1. ``fitness._vmap`` batch evaluation (tests ``jax.vmap`` + ``jax.jit`` on the
   autofit ``Fitness`` wrapper).
2. ``jax.jit(analysis.fit_from)`` round-trip, which relies on the pytree
   registration added to
   ``AnalysisInterferometer._register_fit_interferometer_pytrees`` — this path
   exercises the full ``FitInterferometer`` return value flattening.
"""

import time
from os import path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autogalaxy as ag


dataset_path = path.join("dataset", "interferometer", "jax_test")

if not path.exists(path.join(dataset_path, "data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/interferometer/simulator.py"],
        check=True,
    )

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=3.0,
)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)

print(f"Total Visibilities: {dataset.uv_wavelengths.shape[0]}")

"""
__Model__

Single galaxy with a Sersic bulge — no lens/source split, no mass profile.
"""
bulge = af.Model(ag.lp.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

analysis = ag.AnalysisInterferometer(dataset=dataset)

"""
__vmap Path__

Wrap the autofit ``Fitness`` in ``jax.vmap`` and evaluate a batch of parameter
vectors. This tests that the full likelihood pipeline JIT-compiles end to end.
"""
from autofit.non_linear.fitness import Fitness

batch_size = 50

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters = np.zeros((batch_size, model.total_free_parameters))
for i in range(batch_size):
    parameters[i, :] = model.physical_values_from_prior_medians
parameters = jnp.array(parameters)

start = time.time()
result = fitness._vmap(parameters)
print(result)
print("JAX Time To VMAP + JIT Function:", time.time() - start)

start = time.time()
result = fitness._vmap(parameters)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

"""
__Path A: jit-wrap ``analysis.fit_from``__

Assert that ``jax.jit(analysis.fit_from)(instance)`` returns a
``FitInterferometer`` with a ``jax.Array`` ``log_likelihood`` matching the
NumPy-path scalar. This is the part unblocked by
``_register_fit_interferometer_pytrees``.
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(model)

instance = model.instance_from_prior_medians()

analysis_np = ag.AnalysisInterferometer(dataset=dataset, use_jax=False)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = ag.AnalysisInterferometer(dataset=dataset, use_jax=True)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(fit.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit.log_likelihood)}"
)
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
