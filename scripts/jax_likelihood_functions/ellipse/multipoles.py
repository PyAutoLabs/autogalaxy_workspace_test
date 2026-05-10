"""
Numpy Likelihood: Ellipse Fit With Multipole
=============================================

Step 2 of ``z_features/ellipse_fitting_jax.md`` — multipole variant.

Locks the multipole code path as a numpy-baseline. The JAX-incompatible
``while`` loops in ``EllipseMultipole.get_shape_angle`` are handled by
prompt 5; this script establishes the reference numbers that prompt 7 will
JIT-assert against to ``rtol=1e-4``.

Prints the numpy-path log-likelihood, chi-squared, noise-normalisation, and
figure-of-merit values for a single ellipse + m=4 multipole fit to the
simulated dataset.
"""

from os import path

import autofit as af
import autogalaxy as ag


dataset_path = path.join("dataset", "ellipse", "jax_test")

if not path.exists(path.join(dataset_path, "data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/ellipse/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

"""
__Mask__
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model__
"""
ellipse = af.Model(ag.Ellipse)

ellipse.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
ellipse.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

ellipse.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)
ellipse.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.6, upper_limit=0.6)

ellipse.major_axis = 0.5

multipole = af.Model(ag.EllipseMultipole)
multipole.m = 4
multipole.multipole_comps.multipole_comps_0 = 0.05
multipole.multipole_comps.multipole_comps_1 = 0.0

model = af.Collection(
    ellipses=[ellipse],
    multipoles=[[multipole]],
)

print(model.info)

"""
__Analysis (NumPy Path)__
"""
analysis = ag.AnalysisEllipse(dataset=dataset)  # use_jax defaults to False

instance = model.instance_from_prior_medians()

fit_list = analysis.fit_list_from(instance=instance)

"""
__Reference Numbers__
"""
for i, fit in enumerate(fit_list):
    print(f"Ellipse {i}:")
    print(f"  log_likelihood     = {fit.log_likelihood:.8f}")
    print(f"  chi_squared        = {fit.chi_squared:.8f}")
    print(f"  noise_normalization= {fit.noise_normalization:.8f}")
    print(f"  figure_of_merit    = {fit.figure_of_merit:.8f}")

total_log_likelihood = sum(fit.log_likelihood for fit in fit_list)
total_figure_of_merit = sum(fit.figure_of_merit for fit in fit_list)

print(f"Aggregate:")
print(f"  total_log_likelihood = {total_log_likelihood:.8f}")
print(f"  total_figure_of_merit= {total_figure_of_merit:.8f}")

"""
__TODO(7_analysis_ellipse_jax.md)__

Once `AnalysisEllipse` gains the `use_jax: bool = True` flag and a
`_register_fit_ellipse_pytrees()` helper, this script should additionally:

    analysis_jit = ag.AnalysisEllipse(dataset=dataset, use_jax=True)
    fit_jit_fn = jax.jit(analysis_jit.fit_from)
    fit_jit = fit_jit_fn(instance)

    np.testing.assert_allclose(
        float(fit_jit.log_likelihood),
        total_log_likelihood,
        rtol=1e-4,
    )
"""
