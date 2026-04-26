"""
JAX Likelihood: Delaunay Adapt-Image Pixelization
==================================================

Single-galaxy autogalaxy model using a ``Delaunay`` mesh with a Hilbert
image-mesh (which seeds source-pixel centres in the image plane via an adapt
image) and ``AdaptSplit`` regularization.

This exercises the second post-unflatten lookup site —
``GalaxiesToInversion.image_plane_mesh_grid_list`` — which previously fell
back to the single-mesh-grid value when the by-instance lookup missed.

Two paths are exercised:

1. ``fitness._vmap`` batch evaluation.
2. ``jax.jit(analysis.fit_from)`` scalar round-trip — relies on
   ``AnalysisImaging._register_fit_imaging_pytrees`` and on
   ``AdaptImages.image_plane_mesh_grid_for_galaxy`` resolving fresh-Galaxy
   lookups via the path-tuple list.
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

mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=4,
    over_sample_size_pixelization=4,
)

"""
__JAX & Preloads__

JAX requires static-shaped arrays. ``pixels`` and ``edge_pixels_total`` fix the
total source-pixel count up front. The image-plane mesh grid is built in
NumPy via the Hilbert image-mesh and circle-edge augmentation, then passed
in via ``galaxy_name_image_plane_mesh_grid_dict``.
"""
pixels = 750
edge_pixels_total = 30

galaxy_name_image_dict = {
    "('galaxies', 'galaxy')": dataset.data,
}

image_mesh = ag.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_name_image_dict["('galaxies', 'galaxy')"]
)

image_plane_mesh_grid = ag.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

adapt_images = ag.AdaptImages(
    galaxy_name_image_dict=galaxy_name_image_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'galaxy')": image_plane_mesh_grid
    },
)

"""
__Model__

Single galaxy with a Delaunay pixelization seeded by the Hilbert image-mesh.
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=ag.reg.AdaptSplit,
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

settings = ag.Settings(
    use_border_relocator=True,
    use_positive_only_solver=True,
    use_mixed_precision=True,
)

analysis = ag.AnalysisImaging(
    dataset=dataset, adapt_images=adapt_images, settings=settings
)

"""
__vmap Path__
"""
from autofit.non_linear.fitness import Fitness

batch_size = 3

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
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(model)

instance = model.instance_from_prior_medians()

analysis_np = ag.AnalysisImaging(
    dataset=dataset, adapt_images=adapt_images, settings=settings, use_jax=False
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = ag.AnalysisImaging(
    dataset=dataset, adapt_images=adapt_images, settings=settings, use_jax=True
)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(fit.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit.log_likelihood)}"
)
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-2
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
