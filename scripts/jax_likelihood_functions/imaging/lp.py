"""
JAX Likelihood: Parametric Light Profile
========================================

Verify that JAX can compute the log-likelihood of an ``Imaging`` fit for an
autogalaxy model composed of a linear Sersic bulge. Two paths are exercised:

1. ``fitness._vmap`` batch evaluation (tests ``jax.vmap`` + ``jax.jit`` on the
   autofit ``Fitness`` wrapper).
2. ``jax.jit(analysis.fit_from)`` round-trip, which relies on the pytree
   registration added to ``AnalysisImaging._register_fit_imaging_pytrees`` —
   this path exercises the full ``FitImaging`` return value flattening.
"""

import time
from os import path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autogalaxy as ag


dataset_path = path.join("dataset", "imaging", "jax_test")

if not path.exists(path.join(dataset_path, "data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Model__

Single galaxy with a linear Sersic bulge — no lens/source split, no mass profile.
"""
bulge = af.Model(ag.lp.Sersic)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

analysis = ag.AnalysisImaging(dataset=dataset)

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

Assert that ``jax.jit(analysis.fit_from)(instance)`` returns a ``FitImaging``
with a ``jax.Array`` ``log_likelihood`` matching the NumPy-path scalar. This
is the part unblocked by ``_register_fit_imaging_pytrees``.
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(model)

instance = model.instance_from_prior_medians()

analysis_np = ag.AnalysisImaging(dataset=dataset, use_jax=False)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = ag.AnalysisImaging(dataset=dataset, use_jax=True)
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
