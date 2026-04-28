"""
JAX Likelihood: Rectangular Adapt-Image Pixelization + MGE Bulge (Multi-Wavelength)
=====================================================================================

Verify that JAX can compute the log-likelihood of a multi-wavelength
``Imaging`` fit for an autogalaxy model with two galaxies:
  - galaxy_0: MGE bulge (provides linear light profiles)
  - galaxy_1: adapt-image rectangular pixelization + Constant regularization

Two paths are exercised:

1. ``fitness._vmap`` batch evaluation over a ``af.FactorGraphModel`` that
   combines per-band ``AnalysisImaging`` factors.
2. ``jax.jit`` over a parameter-vector entry point:
   ``instance_from_vector`` → ``factor_graph.log_likelihood_function``.

Uses **option B** — per-band ``galaxy_1.pixelization.regularization.inner_coefficient``
priors via ``model.copy()`` + ``af.GaussianPrior`` on each ``AnalysisFactor``.
The MGE bulge parameters stay shared across bands.

Path A asserts JIT round-trip parity with the vmap result (pixelized path
differs between use_jax=True and use_jax=False).
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
    dataset.apply_over_sampling(
        over_sample_size_lp=1, over_sample_size_pixelization=1
    )
    for dataset in dataset_list
]

"""
__Mesh & Adapt Images (per band)__

The model has two galaxies named ``galaxy_0`` (MGE bulge) and ``galaxy_1``
(pixelization). Both are included in the dict to keep the path list aligned
with all galaxies in the analysis.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

adapt_images_list = [
    ag.AdaptImages(
        galaxy_name_image_dict={
            "('galaxies', 'galaxy_0')": dataset.data,
            "('galaxies', 'galaxy_1')": dataset.data,
        }
    )
    for dataset in dataset_list
]

"""
__Model__

galaxy_0: MGE bulge — provides linear light profiles.
galaxy_1: rectangular adapt-image pixelization.
"""
bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=True
)

galaxy_0 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0),
    regularization=ag.reg.Adapt,
)

galaxy_1 = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(
    galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1)
)

print(model.info)

"""
__Per-band models (option B)__

Each band gets its own ``model.copy()`` with an independent prior on
``galaxy_1.pixelization.regularization.inner_coefficient``.
"""
model_per_band_list = []
for _ in waveband_list:
    model_analysis = model.copy()
    model_analysis.galaxies.galaxy_1.pixelization.regularization.inner_coefficient = (
        af.GaussianPrior(mean=1.0, sigma=0.5)
    )
    model_per_band_list.append(model_analysis)

"""
__FactorGraphModel (vmap path)__
"""
settings = ag.Settings(use_border_relocator=True)

analysis_list = [
    ag.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        settings=settings,
    )
    for dataset, adapt_images in zip(dataset_list, adapt_images_list)
]

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

``FactorGraphModel`` has no ``fit_from`` method, so Path A jit-wraps a
parameter-vector entry point that mirrors what ``fitness._vmap`` does
internally: ``instance_from_vector`` → ``log_likelihood_function``.

The adapt-regularization linear solve has ~1% NumPy/JAX float-ordering
drift — same as the single-dataset autogalaxy ``imaging/rectangular_mge.py``
and ``interferometer/rectangular_mge.py``, so the rtol=1e-2 convention
applies.
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(factor_graph.global_prior_model)

analysis_np_list = [
    ag.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        settings=settings,
        use_jax=False,
    )
    for dataset, adapt_images in zip(dataset_list, adapt_images_list)
]
analysis_factor_np_list = [
    af.AnalysisFactor(prior_model=m, analysis=a)
    for m, a in zip(model_per_band_list, analysis_np_list)
]
factor_graph_np = af.FactorGraphModel(*analysis_factor_np_list, use_jax=False)

params_np = np.array(
    factor_graph_np.global_prior_model.physical_values_from_prior_medians
)
instance_np = factor_graph_np.global_prior_model.instance_from_vector(
    vector=params_np, xp=np
)
log_l_np = float(factor_graph_np.log_likelihood_function(instance_np))
print("NumPy log_likelihood_function:", log_l_np)


@jax.jit
def log_l_jit_fn(parameters):
    instance = factor_graph.global_prior_model.instance_from_vector(
        vector=parameters, xp=jnp
    )
    return factor_graph.log_likelihood_function(instance)


params_jit = jnp.array(
    factor_graph.global_prior_model.physical_values_from_prior_medians
)
log_l_jit = log_l_jit_fn(params_jit)

print("JIT log_likelihood_function:", log_l_jit)
assert isinstance(log_l_jit, jnp.ndarray), (
    f"expected jax.Array, got {type(log_l_jit)}"
)
np.testing.assert_allclose(float(log_l_jit), log_l_np, rtol=1e-2)
print("PASS: jit(log_likelihood_function) round-trip matches NumPy scalar.")
