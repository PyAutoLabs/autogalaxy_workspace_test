"""
Modeling: Single Galaxy Sersic Fit
==================================

End-to-end imaging model-fit on the autogalaxy single-galaxy ``jax_test`` dataset.
Exercises ``AnalysisImaging`` -> ``FitImaging`` with a Nautilus search.

Galaxy: a single ``Sersic`` bulge — no lens / mass / source split (this is autogalaxy,
not autolens).
"""

import os
from os import path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt


"""
__Dataset__

Reuse the ``jax_test`` dataset already used by ``scripts/jax_likelihood_functions/imaging``.
"""
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

aplt.plot_array(array=dataset.data)


"""
__Mask__
"""
mask_radius = 3.0

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.plot_array(array=dataset.data)


"""
__Model__

Single galaxy with a parametric ``Sersic`` bulge.
"""
bulge = af.Model(ag.lp.Sersic)
bulge.centre.centre_0 = 0.0
bulge.centre.centre_1 = 0.0

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


"""
__Search__
"""
search = af.Nautilus(
    path_prefix=path.join("build", "model_fit", "imaging"),
    n_live=50,
    n_like_max=300,
    number_of_cores=2,
)


"""
__Analysis__
"""
analysis = ag.AnalysisImaging(dataset=dataset)


"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)


"""
__Result__
"""
print(result.max_log_likelihood_instance)

aplt.subplot_galaxies(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

aplt.corner_cornerpy(samples=result.samples)
