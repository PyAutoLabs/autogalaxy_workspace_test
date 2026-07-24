"""
Numpy Likelihood: Ellipse Fit
==============================

Step 2 of ``z_features/ellipse_fitting_jax.md``.

Prints the numpy-path reference numbers that prompt 7 will JIT-assert against
to ``rtol=1e-4``. Run this script to capture the baseline log-likelihood,
chi-squared, noise-normalisation, and figure-of-merit values for a single
ellipse fit to the simulated dataset.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

from os import path

import numpy as np

import autofit as af
import autogalaxy as ag


dataset_path = path.join("dataset", "ellipse", "jax_test")

if not path.exists(path.join(dataset_path, "data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/ellipse/jax_likelihood/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

"""
__Mask__
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model__
"""
ellipse = af.Model(ag.Ellipse)

ellipse.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
ellipse.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

ellipse.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)
ellipse.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)

ellipse.major_axis = 0.5

model = af.Collection(ellipses=[ellipse])

print(model.info)

"""
__Analysis (NumPy Path)__
"""
analysis = ag.AnalysisEllipse(dataset=dataset, use_jax=False)

instance = model.instance_from_prior_medians()

fit_list = analysis.fit_list_from(instance=instance)

"""
__Reference Numbers__
"""
for i, fit in enumerate(fit_list):
    print(f"Ellipse {i}:")
    print(f"  log_likelihood     = {fit.log_likelihood:.8f}")
    print(f"  chi_squared        = {fit.chi_squared:.8f}")
    print(f"  noise_normalization= {fit.noise_normalization:.8f}")
    print(f"  figure_of_merit    = {fit.figure_of_merit:.8f}")

total_log_likelihood = sum(fit.log_likelihood for fit in fit_list)
total_figure_of_merit = sum(fit.figure_of_merit for fit in fit_list)

print(f"Aggregate:")
print(f"  total_log_likelihood = {total_log_likelihood:.8f}")
print(f"  total_figure_of_merit= {total_figure_of_merit:.8f}")

"""
__vmap Path__

Wrap the autofit ``Fitness`` in ``jax.vmap`` and evaluate a batch of parameter
vectors. This exercises the full likelihood pipeline through JIT.
"""
import time
import jax
import jax.numpy as jnp
from autofit.non_linear.fitness import Fitness

batch_size = 50

fitness = Fitness(
    model=model,
    analysis=ag.AnalysisEllipse(dataset=dataset, use_jax=True),
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
__JIT fit_from round-trip__

Assert that ``jax.jit(analysis.fit_from)(instance)`` returns a ``FitEllipseSummed``
with a ``jax.Array`` ``log_likelihood`` matching the NumPy-path scalar.
"""


analysis_jit = ag.AnalysisEllipse(dataset=dataset, use_jax=True)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(
    fit.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit.log_likelihood)}"
np.testing.assert_allclose(float(fit.log_likelihood), total_log_likelihood, rtol=1e-4)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
