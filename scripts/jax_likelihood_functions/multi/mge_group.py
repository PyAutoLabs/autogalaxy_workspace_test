"""
JAX Likelihood: MGE Basis Light Profile with Extra Galaxies (Multi-Wavelength)
===============================================================================

Verify that JAX can compute the log-likelihood of a multi-wavelength
``Imaging`` fit for an autogalaxy model composed of an MGE linear basis on the
primary galaxy plus extra galaxies (each with a spherical MGE linear basis).
Two paths are exercised:

1. ``fitness._vmap`` batch evaluation over a ``af.FactorGraphModel`` that
   combines per-band ``AnalysisImaging`` factors.
2. ``jax.jit`` over a parameter-vector entry point:
   ``instance_from_vector`` → ``factor_graph.log_likelihood_function``.

Uses **option B** — per-band ``galaxy.bulge`` MGE ``ell_comps`` priors via
``model.copy()`` + ``af.GaussianPrior`` on each ``AnalysisFactor``. The extra
galaxies stay shared across bands.
"""

from os import path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autogalaxy as ag


waveband_list = ["g", "r"]
pixel_scales = 0.1
mask_radius = 3.0

dataset_path = path.join("dataset", "multi", "jax_test")

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

"""
__Group Centres__

The multi simulator does not include extra galaxies, so the extra-galaxy
components here have no data support. They still exercise the MGE +
``extra_galaxies`` wiring through the JAX factor graph.
"""
centre_list = [(0.0, 1.0), (1.0, 0.0)]

"""
__Model__

Primary galaxy with a large MGE bulge. Extra galaxies with spherical MGE
bases. No mass profiles, no lens/source split.
"""
total_gaussians = 20
gaussian_per_basis = 2
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []
for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
    )
    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]
    bulge_gaussian_list += gaussian_list

bulge = af.Model(ag.lp_basis.Basis, profile_list=bulge_gaussian_list)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

# Extra Galaxies:
extra_galaxies_list = []

for extra_galaxy_centre in centre_list:
    total_gaussians = 8
    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    eg_gaussian_list = af.Collection(
        af.Model(ag.lp_linear.GaussianSph) for _ in range(total_gaussians)
    )
    for i, gaussian in enumerate(eg_gaussian_list):
        gaussian.centre.centre_0 = extra_galaxy_centre[0]
        gaussian.centre.centre_1 = extra_galaxy_centre[1]
        gaussian.sigma = 10 ** log10_sigma_list[i]

    extra_galaxy_bulge = af.Model(ag.lp_basis.Basis, profile_list=eg_gaussian_list)

    extra_galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge)
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

model = af.Collection(
    galaxies=af.Collection(galaxy=galaxy), extra_galaxies=extra_galaxies
)

print(model.info)

"""
__Per-band models (option B)__

Each band gets its own ``model.copy()`` with independent primary-galaxy MGE
``ell_comps`` priors. The extra galaxies stay shared.
"""
model_per_band_list = []
for _ in waveband_list:
    model_analysis = model.copy()
    ec_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    ec_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    for gaussian in model_analysis.galaxies.galaxy.bulge.profile_list:
        gaussian.ell_comps.ell_comps_0 = ec_0
        gaussian.ell_comps.ell_comps_1 = ec_1
    model_per_band_list.append(model_analysis)

"""
__FactorGraphModel (vmap path)__
"""
analysis_list = [ag.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=m, analysis=analysis)
    for m, analysis in zip(model_per_band_list, analysis_list)
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

print(factor_graph.global_prior_model.info)

from autofit.non_linear.fitness import Fitness
import time

batch_size = 3

fitness = Fitness(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters = np.zeros(
    (batch_size, factor_graph.global_prior_model.total_free_parameters)
)
for i in range(batch_size):
    parameters[i, :] = (
        factor_graph.global_prior_model.physical_values_from_prior_medians
    )
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
__Path A: jit-wrap ``factor_graph.log_likelihood_function``__
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(factor_graph.global_prior_model)

analysis_np_list = [
    ag.AnalysisImaging(dataset=dataset, use_jax=False) for dataset in dataset_list
]
factor_graph_np = af.FactorGraphModel(
    *[
        af.AnalysisFactor(prior_model=m, analysis=a)
        for m, a in zip(model_per_band_list, analysis_np_list)
    ],
    use_jax=False,
)

params_np = np.array(
    factor_graph_np.global_prior_model.physical_values_from_prior_medians
)
instance_np = factor_graph_np.global_prior_model.instance_from_vector(
    vector=params_np, xp=np
)
log_l_np = float(factor_graph_np.log_likelihood_function(instance_np))
print("NumPy log_likelihood_function:", log_l_np)

analysis_jit_list = [
    ag.AnalysisImaging(dataset=dataset, use_jax=True) for dataset in dataset_list
]
factor_graph_jit = af.FactorGraphModel(
    *[
        af.AnalysisFactor(prior_model=m, analysis=a)
        for m, a in zip(model_per_band_list, analysis_jit_list)
    ],
    use_jax=True,
)


@jax.jit
def log_l_jit_fn(parameters):
    instance = factor_graph_jit.global_prior_model.instance_from_vector(
        vector=parameters, xp=jnp
    )
    return factor_graph_jit.log_likelihood_function(instance)


params_jit = jnp.array(
    factor_graph_jit.global_prior_model.physical_values_from_prior_medians
)
log_l_jit = log_l_jit_fn(params_jit)

print("JIT log_likelihood_function:", log_l_jit)
assert isinstance(log_l_jit, jnp.ndarray), (
    f"expected jax.Array, got {type(log_l_jit)}"
)
np.testing.assert_allclose(float(log_l_jit), log_l_np, rtol=1e-4)
print("PASS: jit(log_likelihood_function) round-trip matches NumPy scalar.")
