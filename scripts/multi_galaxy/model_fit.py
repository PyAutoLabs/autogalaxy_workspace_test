"""
Modeling: Multi Galaxy Model Fit
================================

Integration test of the multi-galaxy regime introduced in `autogalaxy_workspace/scripts/multi_galaxy/`:
two blended galaxies, **one free light model per galaxy**, fitted simultaneously with the standard
``AnalysisImaging`` workflow. Simulates its own dataset inline so it is fully self-contained.

Structural assertions lock in the regime: independent per-galaxy light models (no shared priors between
the two galaxies), composed with the list-based ``galaxy_0`` / ``galaxy_1`` API.
"""

import numpy as np
from os import path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Simulate__

A close blended pair (~1.5" separation, comparable brightness).
"""
grid = ag.Grid2D.uniform(shape_native=(120, 120), pixel_scales=0.1)

psf = ag.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
)

simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

galaxy_centres = [(0.0, -0.75), (0.0, 0.75)]

galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.SersicSph(
        centre=galaxy_centres[0], intensity=1.0, effective_radius=0.8, sersic_index=3.0
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.SersicSph(
        centre=galaxy_centres[1], intensity=0.7, effective_radius=1.0, sersic_index=1.5
    ),
)

dataset = simulator.via_galaxies_from(
    galaxies=ag.Galaxies(galaxies=[galaxy_0, galaxy_1]), grid=grid
)

"""
__Mask__

Must enclose BOTH galaxies' blended light.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

aplt.plot_array(array=dataset.data)

"""
__Model__

One free ``SersicSph`` per galaxy, list-based composition.
"""
galaxy_dict = {}

for i, centre in enumerate(galaxy_centres):
    bulge = af.Model(ag.lp.SersicSph)
    bulge.centre = centre

    galaxy_dict[f"galaxy_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

"""
__Model Structure Assertions__

Each galaxy contributes its own independent free light model.
"""
assert galaxy_dict["galaxy_0"].prior_count == galaxy_dict["galaxy_1"].prior_count == 3
assert model.prior_count == 6  # fully independent — no shared priors between the pair
assert model.info.count("intensity") >= 2

"""
__Search + Analysis + Fit__
"""
search = af.Nautilus(
    path_prefix=path.join("build", "model_fit", "multi_galaxy"),
    n_live=50,
    n_like_max=300,
    number_of_cores=2,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

Two galaxies come back, each with its own fitted light model.
"""
galaxies = result.max_log_likelihood_galaxies

assert len(galaxies) == 2

for i, galaxy in enumerate(galaxies):
    print(f"galaxy_{i}: centre = {galaxy.bulge.centre}")

aplt.subplot_galaxies(galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)
