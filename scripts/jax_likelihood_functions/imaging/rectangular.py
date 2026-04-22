"""
JAX Likelihood: Rectangular Pixelization
=========================================

Verify that JAX can compute the log-likelihood of an ``Imaging`` fit for an
autogalaxy model using a non-adapt rectangular pixelization mesh. Two paths
are exercised:

1. ``fitness._vmap`` batch evaluation.
2. ``jax.jit(analysis.fit_from)`` scalar round-trip — relies on
   ``AnalysisImaging._register_fit_imaging_pytrees``.

Note: this port uses ``ag.mesh.RectangularUniform`` + ``ag.reg.Constant`` (no
adapt images). The adapt-image variant (``RectangularAdaptImage`` +
``ag.reg.Adapt``) hits a post-unflatten Galaxy-identity mismatch in
``AdaptImages.galaxy_image_dict`` that the autogalaxy library does not yet
resolve across the JIT boundary — a separate library fix is required there.
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

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

"""
__Model__

Single galaxy with a rectangular pixelization. No lens/source split, no mass
profile, no adapt images.
"""
mesh = ag.mesh.RectangularUniform(shape=(28, 28))
regularization = ag.reg.Constant(coefficient=1.0)
pixelization = ag.Pixelization(mesh=mesh, regularization=regularization)

galaxy = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

print(model.info)

analysis = ag.AnalysisImaging(dataset=dataset)

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

analysis_np = ag.AnalysisImaging(dataset=dataset, use_jax=False)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = ag.AnalysisImaging(dataset=dataset, use_jax=True)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(fit.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit.log_likelihood)}"
)
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
