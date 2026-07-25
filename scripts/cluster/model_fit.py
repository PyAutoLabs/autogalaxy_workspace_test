"""
Modeling: Cluster Model Fit
===========================

Integration test of the cluster regime introduced in `autogalaxy_workspace/scripts/cluster/`: a BCG
modeled individually plus a catalogue-driven member tier whose intensities are tied to catalogue
luminosities through one shared free normalization. Simulates its own field inline (1 BCG + 6 members)
so it is fully self-contained.

Structural assertions lock in the regime: the member tier contributes exactly ONE free parameter
regardless of population size (prior-arithmetic tie ``intensity = scale * luminosity``), and the whole
fit runs through the standard ``AnalysisImaging`` — in autogalaxy the foreground galaxies' light IS the
subject (the deliberate divergence from autolens's point-source cluster workflow).
"""

import numpy as np
from os import path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Simulate__

A BCG + 6 members whose intensities equal their catalogue luminosities (true shared scale = 1).
"""
grid = ag.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.1)

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

bcg_centre = (0.0, 0.0)
member_centres = [(3.5, -4.0), (-4.5, 2.0), (2.0, 5.0), (5.0, 3.0), (-4.0, -5.0), (-1.5, 4.5)]
member_luminosities = [0.40, 0.30, 0.22, 0.16, 0.10, 0.06]

bcg = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=bcg_centre,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        intensity=1.5,
        effective_radius=2.0,
        sersic_index=4.0,
    ),
)

members = [
    ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.SersicSph(
            centre=centre, intensity=lum, effective_radius=0.6, sersic_index=3.0
        ),
    )
    for centre, lum in zip(member_centres, member_luminosities)
]

dataset = simulator.via_galaxies_from(
    galaxies=ag.Galaxies(galaxies=[bcg] + members), grid=grid
)

"""
__Mask__
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=7.0,
)

dataset = dataset.apply_mask(mask=mask)

aplt.plot_array(array=dataset.data)

"""
__Model__

BCG: free Sersic. Members: fixed centres/shapes, intensities tied to catalogue luminosities via one
shared free ``intensity_scale``.
"""
bcg_bulge = af.Model(ag.lp.Sersic)
bcg_bulge.centre = bcg_centre

galaxy_dict = {"bcg": af.Model(ag.Galaxy, redshift=0.5, bulge=bcg_bulge)}

intensity_scale = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

for i, (centre, lum) in enumerate(zip(member_centres, member_luminosities)):
    bulge = af.Model(ag.lp.SersicSph)
    bulge.centre = centre
    bulge.intensity = intensity_scale * lum
    bulge.effective_radius = 0.6
    bulge.sersic_index = 3.0

    galaxy_dict[f"member_{i}"] = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**galaxy_dict))

"""
__Model Structure Assertions__

The member tier contributes exactly ONE free parameter regardless of how many members it holds.
"""
bcg_only = af.Collection(galaxies=af.Collection(bcg=galaxy_dict["bcg"]))

assert model.prior_count == bcg_only.prior_count + 1  # the entire tier = 1 parameter

"""
__Search + Analysis + Fit__
"""
search = af.Nautilus(
    path_prefix=path.join("build", "model_fit", "cluster"),
    n_live=50,
    n_like_max=300,
    number_of_cores=2,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

All 7 galaxies come back; every member's fitted intensity sits exactly on the shared scale times its
catalogue luminosity.
"""
instance = result.max_log_likelihood_instance

galaxies = result.max_log_likelihood_galaxies

assert len(galaxies) == 7

scale_0 = instance.galaxies.member_0.bulge.intensity / member_luminosities[0]
for i, lum in enumerate(member_luminosities):
    scale_i = getattr(instance.galaxies, f"member_{i}").bulge.intensity / lum
    assert np.isclose(float(scale_i), float(scale_0), rtol=1e-8)

print(f"member tier intensity_scale = {float(scale_0):.3f}")

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)
