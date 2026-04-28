"""
JAX Likelihood: Delaunay Adapt-Image Pixelization (Interferometer)
===================================================================

Single-galaxy autogalaxy model using a ``Delaunay`` mesh with a Hilbert
image-mesh (which seeds source-pixel centres in the image plane via an adapt
image) and ``AdaptSplit`` regularization.

This exercises the second post-unflatten lookup site —
``GalaxiesToInversion.image_plane_mesh_grid_list`` — which previously fell
back to the single-mesh-grid value when the by-instance lookup missed.

Two paths are exercised:

1. ``fitness._vmap`` batch evaluation.
2. ``jax.jit(analysis.fit_from)`` scalar round-trip — relies on
   ``AnalysisInterferometer._register_fit_interferometer_pytrees`` and on
   ``AdaptImages.image_plane_mesh_grid_for_galaxy`` resolving fresh-Galaxy
   lookups via the path-tuple list.

Note: interferometer does not use over-sampling — no ``apply_over_sampling``
calls appear here.
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
__JAX & Preloads__

JAX requires static-shaped arrays. ``pixels`` and ``edge_pixels_total`` fix the
total source-pixel count up front. The image-plane mesh grid is built in
NumPy via the Hilbert image-mesh and circle-edge augmentation, then passed
in via ``galaxy_name_image_plane_mesh_grid_dict``.
"""
mask_radius = 3.0
pixels = 750
edge_pixels_total = 30

# Use a Sersic image as adapt data to avoid negative values in the dirty image
# causing NaN in the pixel signal computation (sqrt of negative signal).
bulge_adapt = ag.lp.Sersic()
adapt_image = bulge_adapt.image_2d_from(grid=dataset.grid)

galaxy_name_image_dict = {
    "('galaxies', 'galaxy')": adapt_image,
}

image_mesh = ag.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=real_space_mask, adapt_data=galaxy_name_image_dict["('galaxies', 'galaxy')"]
)

image_plane_mesh_grid = ag.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=real_space_mask.mask_centre,
    radius=mask_radius + real_space_mask.pixel_scale / 2.0,
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

analysis = ag.AnalysisInterferometer(
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

analysis_np = ag.AnalysisInterferometer(
    dataset=dataset, adapt_images=adapt_images, settings=settings, use_jax=False
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = ag.AnalysisInterferometer(
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
