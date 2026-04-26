"""
JAX Likelihood: Rectangular Adapt-Image Pixelization + MGE Bulge
=================================================================

Two-galaxy autogalaxy model: a foreground galaxy with an MGE bulge and a
second galaxy with an adapt-image rectangular pixelization
(``RectangularAdaptImage`` + ``Constant`` regularization).

This is the multi-pixelization regression case the path-tuple library fix
was made for: prior to the fix the autolens fallback would silently return
the wrong adapt image when more than one galaxy is present.

Two paths are exercised:

1. ``fitness._vmap`` batch evaluation.
2. ``jax.jit(analysis.fit_from)`` scalar round-trip — relies on
   ``AnalysisImaging._register_fit_imaging_pytrees`` and on
   ``AdaptImages.image_for_galaxy`` resolving fresh-Galaxy lookups via the
   path-tuple list.
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

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)

"""
__Adapt Images__

The model has two galaxies named ``galaxy_0`` (MGE bulge) and ``galaxy_1``
(pixelization). ``galaxy_1`` is the only one that needs an adapt image, but
``galaxy_0`` is included in the dict to keep the path list aligned with all
galaxies in the analysis.
"""
galaxy_name_image_dict = {
    "('galaxies', 'galaxy_0')": dataset.data,
    "('galaxies', 'galaxy_1')": dataset.data,
}

adapt_images = ag.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

"""
__Model__

galaxy_0: MGE bulge — provides linear light profiles.
galaxy_1: rectangular adapt-image pixelization — exercises the adapt-image
inversion path that was previously broken across the JIT boundary.
"""
bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    centre_prior_is_uniform=True,
)

galaxy_0 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

mesh = ag.mesh.RectangularAdaptImage(shape=(28, 28))
regularization = ag.reg.Constant(coefficient=1.0)
pixelization = ag.Pixelization(mesh=mesh, regularization=regularization)

galaxy_1 = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(
    galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1)
)

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
