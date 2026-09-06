"""
JAX Likelihood: Multipole Light Profile
========================================

Verify that JAX can compute the log-likelihood of an ``Imaging`` fit for an
autogalaxy model whose galaxy has a ``SersicMultipole`` bulge with m=3 and m=4
Fourier perturbations on the eccentric radius. Two paths are exercised:

1. ``fitness._vmap`` batch evaluation (``jax.vmap`` + ``jax.jit`` on the
   autofit ``Fitness`` wrapper).
2. ``jax.jit(analysis.fit_from)`` round-trip, asserting the JIT scalar
   matches the NumPy-path scalar.

Mirrors ``imaging/lp.py`` but uses ``ag.lp.SersicMultipole`` and sets explicit
Gaussian priors on the four multipole component parameters
(``multipole_3_comps_0``, ``multipole_3_comps_1``, ``multipole_4_comps_0``,
``multipole_4_comps_1``). The library does not yet ship default priors for
those — this keeps the test self-contained until the workspace priors follow-up
lands.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

import time
from os import path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autogalaxy as ag


dataset_path = path.join("dataset", "imaging", "jax_test")

if ag.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/jax_likelihood/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.3,
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

Single galaxy with a SersicMultipole bulge. Multipole component priors are set
inline so all four are free parameters in the fit.
"""
bulge = af.Model(ag.lp.SersicMultipole)
bulge.multipole_3_comps = af.TuplePrior(
    multipole_3_comps_0=af.GaussianPrior(mean=0.0, sigma=0.05),
    multipole_3_comps_1=af.GaussianPrior(mean=0.0, sigma=0.05),
)
bulge.multipole_4_comps = af.TuplePrior(
    multipole_4_comps_0=af.GaussianPrior(mean=0.0, sigma=0.05),
    multipole_4_comps_1=af.GaussianPrior(mean=0.0, sigma=0.05),
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

analysis = ag.AnalysisImaging(dataset=dataset)

"""
__vmap Path__

Wrap the autofit ``Fitness`` in ``jax.vmap`` and evaluate a batch of parameter
vectors. Tests that the full likelihood pipeline JIT-compiles end to end with
the multipole perturbation in the ``image_2d_from`` call chain.
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
with a ``jax.Array`` ``log_likelihood`` matching the NumPy-path scalar.
"""


instance = model.instance_from_prior_medians()

analysis_np = ag.AnalysisImaging(dataset=dataset, use_jax=False)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = ag.AnalysisImaging(dataset=dataset, use_jax=True)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(
    fit.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit.log_likelihood)}"
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
