"""
Simulator: Point-Source Integration Test
========================================

Exercises the standard and linear point-source light profiles introduced by
PyAutoGalaxy PR #562.

The standard `ag.lp.PointSource` is simulated with an over-sampled PSF. Its
`intensity` is total flux, so a point-only noiseless image must retain that flux
after fine-grid convolution and binning back to detector resolution.

The linear `ag.lp_linear.PointSource` has unit flux in its inversion mapping
matrix. Converting it with a solved coefficient must return the corresponding
standard point source with that coefficient as its total flux.
"""

import numpy as np

import autogalaxy as ag

over_sample_size = 2
pixel_scale = 0.1
point_source_flux = 25.0
point_source_centre = (0.025, -0.025)

grid = ag.Grid2D.uniform(
    shape_native=(51, 51),
    pixel_scales=pixel_scale,
    over_sample_size=over_sample_size,
)

psf = ag.Convolver.from_gaussian(
    shape_native=(21, 21),
    sigma=0.12,
    pixel_scales=grid.pixel_scales[0] / over_sample_size,
    normalize=True,
    convolve_over_sample_size=over_sample_size,
)

simulator = ag.SimulatorImaging(
    exposure_time=1.0,
    psf=psf,
    background_sky_level=1.0,
    add_poisson_noise_to_data=False,
)

point_source = ag.lp.PointSource(
    centre=point_source_centre,
    intensity=point_source_flux,
)

galaxy = ag.Galaxy(redshift=0.5, point_source=point_source)
galaxies = ag.Galaxies(galaxies=[galaxy])

dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

np.testing.assert_allclose(
    np.sum(dataset.data), point_source_flux, rtol=1.0e-10, atol=1.0e-10
)

linear_point_source = ag.lp_linear.PointSource(centre=point_source_centre)
converted_point_source = linear_point_source.lp_instance_from(
    linear_light_profile_intensity_dict={linear_point_source: point_source_flux}
)

assert type(converted_point_source) is ag.lp.PointSource
assert converted_point_source.centre == point_source_centre
assert converted_point_source.intensity == point_source_flux

print(
    "Point-source simulation conserved total flux and linear conversion passed: ",
    point_source_flux,
)
