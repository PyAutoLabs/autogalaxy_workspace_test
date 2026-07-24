"""
Tests that jax.value_and_grad can compute finite, non-NaN gradients of the log-likelihood
for an autogalaxy interferometer model with a Multi-Gaussian Expansion (MGE) linear basis
light profile. This tests the core JAX differentiability that enables gradient-based
inference on visibility data.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX gradient scripts exercise jax.value_and_grad on the full likelihood
path; need JAX enabled and full-size datasets.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autogalaxy as ag

dataset_path = path.join("dataset", "interferometer", "jax_test")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(path.join(dataset_path, "data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/jax_likelihood/simulator.py",
        ],
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

# Single galaxy with an MGE linear basis light profile — no lens/source split, no mass profile.

mask_radius = 3.0

bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

analysis = ag.AnalysisInterferometer(dataset=dataset)

from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

# Perturb ell_comps away from (0,0) to avoid degenerate gradients at the
# circular-profile singularity (arctan2 gradient is undefined at exactly (0,0)).
key = jax.random.PRNGKey(0)
perturbation = jax.random.uniform(
    key, shape=param_vector.shape, minval=0.01, maxval=0.05
)
param_vector = param_vector + perturbation

value, grad = jax.value_and_grad(fitness.call)(param_vector)

print(f"Log likelihood = {float(value):.6f}")
print(f"Gradient shape = {grad.shape}")
print(f"Gradient = {np.array(grad)}")

assert np.isfinite(float(value)), "Log likelihood is not finite"
assert grad.shape == (
    model.total_free_parameters,
), f"Gradient shape mismatch: {grad.shape}"
assert np.all(
    np.isfinite(np.array(grad))
), f"Gradient contains non-finite values: {np.array(grad)}"
assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"

print("jax_grad/interferometer/mge.py JAX gradient checks passed.")
