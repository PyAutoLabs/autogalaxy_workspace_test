"""
Tests that jax.value_and_grad can compute finite, non-NaN gradients of the log-likelihood
for an autogalaxy multi-wavelength model with a Multi-Gaussian Expansion (MGE) linear
basis light profile. The gradient is taken over the joint parameter vector spanning both
g and r bands via an af.FactorGraphModel. This tests the core JAX differentiability of
the multi-dataset likelihood path.

Uses option B — per-band MGE ``ell_comps_{0,1}`` priors via ``model.copy()`` + a fresh
``af.GaussianPrior`` pair re-tied across every gaussian within each basis.

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

waveband_list = ["g", "r"]
pixel_scales = 0.1
mask_radius = 3.0

dataset_path = path.join("dataset", "multi", "jax_test")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(path.join(dataset_path, "g_data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/multi/simulator.py"],
        check=True,
    )

dataset_list = [
    ag.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{band}_data.fits"),
        psf_path=path.join(dataset_path, f"{band}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{band}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for band in waveband_list
]

mask_list = [
    ag.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    for dataset in dataset_list
]

dataset_list = [
    dataset.apply_mask(mask=mask) for dataset, mask in zip(dataset_list, mask_list)
]
dataset_list = [
    dataset.apply_over_sampling(over_sample_size_lp=1) for dataset in dataset_list
]

# Single galaxy with an MGE linear basis light profile — no lens/source split, no mass profile.

bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

# Per-band models (option B): re-tie the MGE basis ell_comps to a fresh prior pair
# per band so each band gets its own shape freedom while sharing centres and intensities.

model_per_band_list = []
for _ in waveband_list:
    model_analysis = model.copy()
    ec_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    ec_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    for gaussian in model_analysis.galaxies.galaxy.bulge.profile_list:
        gaussian.ell_comps.ell_comps_0 = ec_0
        gaussian.ell_comps.ell_comps_1 = ec_1
    model_per_band_list.append(model_analysis)

analysis_list = [ag.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=m, analysis=analysis)
    for m, analysis in zip(model_per_band_list, analysis_list)
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(factor_graph.global_prior_model.info)

from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(
    factor_graph.global_prior_model.physical_values_from_prior_medians
)

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
    factor_graph.global_prior_model.total_free_parameters,
), f"Gradient shape mismatch: {grad.shape}"
assert np.all(
    np.isfinite(np.array(grad))
), f"Gradient contains non-finite values: {np.array(grad)}"
assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"

print("jax_grad/multi/mge.py JAX gradient checks passed.")
