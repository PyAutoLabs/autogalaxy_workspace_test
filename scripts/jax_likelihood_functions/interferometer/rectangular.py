"""
JAX Likelihood: Rectangular Adapt-Image Pixelization (Interferometer)
======================================================================

Verify that JAX can compute the log-likelihood of an ``Interferometer`` fit for
an autogalaxy model that uses an adapt-image rectangular pixelization
(``RectangularAdaptImage`` + ``Adapt`` regularization).

Two paths are exercised:

1. ``fitness._vmap`` batch evaluation.
2. ``jax.jit(analysis.fit_from)`` scalar round-trip — relies on
   ``AnalysisInterferometer._register_fit_interferometer_pytrees`` and on
   ``AdaptImages.image_for_galaxy`` resolving fresh-Galaxy lookups via the
   path-tuple list across the JIT boundary.

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
__Adapt Images__

The galaxy is named ``galaxy`` in the model, so the path tuple is
``('galaxies', 'galaxy')``. ``dataset.data`` is used as a stand-in for the
"previous-fit" galaxy image — sufficient to exercise the adapt-image code paths.
"""
galaxy_name_image_dict = {
    "('galaxies', 'galaxy')": dataset.dirty_image,
}

adapt_images = ag.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

"""
__Model__

Single galaxy with an adapt-image rectangular pixelization. The mesh shape is
fixed (28 x 28) per the JAX static-shape requirement.
"""
mesh = ag.mesh.RectangularAdaptImage(shape=(28, 28), weight_power=1.0)
regularization = ag.reg.Adapt()
pixelization = ag.Pixelization(mesh=mesh, regularization=regularization)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

settings = ag.Settings(
    use_border_relocator=True,
    use_positive_only_solver=True,
    use_mixed_precision=True,
)

analysis = ag.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=settings,
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
    dataset=dataset,
    adapt_images=adapt_images,
    settings=settings,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = ag.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    settings=settings,
    use_jax=True,
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
