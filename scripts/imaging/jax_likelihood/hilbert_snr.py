"""
JAX Likelihood: HilbertSNR Image Mesh With Soft S/N Adapt Images
===============================================================

The default adaptive set-up of the workspaces (adapt-mesh-weighting, PyAutoArray#714): a `HilbertSNR` image mesh
drawn from the galaxy's signal-to-noise map, with edge points from `edge_pixels`, and the soft S/N maps of
`soft_snr_maps_from` as the adapt images that `AdaptSplit` adapts to. The galaxy key is written short (`"galaxy"`).

This pins the NumPy figure of merit (log evidence), checks the JAX `vmap` path against it and
`jax.jit(analysis.fit_from)` against the NumPy log likelihood, so the new default stays JAX-compatible.
`imaging/jax_likelihood/delaunay.py` keeps the older `Hilbert` set-up.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""
from os import path

import numpy as np

# Import autogalaxy before jax: it enables JAX 64-bit precision, which only
# takes effect if set before jax is first imported.
import autofit as af
import autogalaxy as ag
import jax
import jax.numpy as jnp

dataset_path = path.join("dataset", "imaging", "jax_test")

ag.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/imaging/jax_likelihood/simulator.py"
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
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=2,
    over_sample_size_pixelization=4,
)

"""
__Adapt Images__

The galaxy is the only emission, so its S/N map is the data divided by the noise map (in a pipeline,
`ag.galaxy_snr_maps_from(result)` or `ag.AdaptImages.from_result` compute it from the previous fit).
"""
snr_map = ag.Array2D(
    values=np.asarray(dataset.data.slim) / np.asarray(dataset.noise_map.slim),
    mask=dataset.mask,
)
snr_maps = {"('galaxies', 'galaxy')": snr_map}

image_mesh = ag.image_mesh.HilbertSNR(pixels=750, edge_pixels=30)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=snr_map
)

assert image_plane_mesh_grid.shape[0] == image_mesh.pixels + image_mesh.edge_pixels

adapt_images = ag.AdaptImages(
    galaxy_name_image_dict=ag.soft_snr_maps_from(snr_maps),
    galaxy_name_image_plane_mesh_grid_dict={"galaxy": image_plane_mesh_grid},
)

"""
__Model__
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.Delaunay(pixels=image_mesh.pixels, zeroed_pixels=image_mesh.edge_pixels),
    regularization=ag.reg.AdaptSplit,
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

settings = ag.Settings(
    use_border_relocator=True,
    use_positive_only_solver=True,
    use_mixed_precision=True,
)

"""
__NumPy Figure of Merit (pinned)__
"""
instance = model.instance_from_prior_medians()

analysis_np = ag.AnalysisImaging(
    dataset=dataset, adapt_images=adapt_images, settings=settings, use_jax=False
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))
print("NumPy fit.figure_of_merit (log evidence):", float(fit_np.figure_of_merit))

# Pinned 2026-11-26 (autolens_workspace#693): HilbertSNR(pixels=750, edge_pixels=30) defaults
# (8-point cap, frame-independent interpolation) on the jax_test dataset, soft S/N adapt images.
np.testing.assert_allclose(
    float(fit_np.figure_of_merit),
    -547.27856589,
    rtol=1e-4,
    err_msg="hilbert_snr: NumPy figure of merit mismatch",
)

"""
__JAX: vmap and jit(fit_from)__

Mixed precision is on, so JAX is compared with NumPy at 1e-2, as in `delaunay.py`.
"""
from autofit.non_linear.fitness import Fitness

analysis = ag.AnalysisImaging(
    dataset=dataset, adapt_images=adapt_images, settings=settings
)

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)
parameters = jnp.array(np.tile(model.physical_values_from_prior_medians, (3, 1)))
result = np.asarray(fitness._vmap(parameters))
print("JAX vmap figure of merit:", result)

# The vmapped fitness returns the figure of merit (the log evidence for a pixelized galaxy).
np.testing.assert_allclose(
    result,
    float(fit_np.figure_of_merit),
    rtol=1e-2,
    err_msg="hilbert_snr: JAX vmap figure of merit differs from NumPy",
)

analysis_jit = ag.AnalysisImaging(
    dataset=dataset, adapt_images=adapt_images, settings=settings, use_jax=True
)
fit = jax.jit(analysis_jit.fit_from)(instance)
print("JIT fit.log_likelihood:", fit.log_likelihood)

np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-2
)
print("PASS: HilbertSNR + soft S/N adapt images agree across NumPy, vmap and jit.")
