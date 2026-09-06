"""
Simulator: JAX Imaging Test Datasets
====================================

Simulates the two `Imaging` datasets consumed by every script in
``scripts/imaging/jax_likelihood/`` (and the ``jax_grad`` / ``visualization``
scripts that re-use them).

A single galaxy with a Sersic bulge + Exponential disk is imaged at ground-based
resolution and signal-to-noise. No lens / mass / source plane — this is a
single-plane autogalaxy dataset designed to exercise the JAX likelihood path on
parametric light profiles, MGE bases, and pixelization sources.

**Two datasets, because the model in each script must match the data it fits.**
The ``mge_group.py`` scripts fit a primary galaxy *plus* extra galaxies; fitting
extra galaxies against data that contains none is a model that does not match
its data. So this simulator writes:

- ``dataset/imaging/jax_test`` — the single galaxy, fit by every other script.
- ``dataset/imaging/jax_test_group`` — the same galaxy plus the two extra
  galaxies the ``mge_group.py`` model actually composes, at the same centres.

Resolution is set by what the models resolve, not by what a telescope could
deliver: a 100 x 100 grid at 0.3"/pixel with a 0.8" FWHM Gaussian PSF
(``sigma=0.35``), which stays well sampled at 0.3"/pixel.

Output files (under each dataset directory):

- ``data.fits`` — the simulated noisy image
- ``psf.fits`` — the Gaussian PSF kernel used during simulation
- ``noise_map.fits`` — per-pixel 1-sigma noise map
- ``galaxies.json`` — the exact ``Galaxies`` used, for reproducibility

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

from pathlib import Path

import autogalaxy as ag
import autogalaxy.plot as aplt


grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.3)

psf = ag.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.35, pixel_scales=grid.pixel_scales, normalize=True
)

simulator = ag.SimulatorImaging(
    exposure_time=2000.0,
    psf=psf,
    background_sky_level=1.0,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=ag.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
        intensity=2.0,
        effective_radius=1.6,
    ),
)

extra_galaxy_centre_list = [(1.2, 1.2), (-1.0, 1.5)]

extra_galaxies = [
    ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.SersicSph(
            centre=centre,
            intensity=1.0,
            effective_radius=0.35,
            sersic_index=2.0,
        ),
    )
    for centre in extra_galaxy_centre_list
]

galaxies_via_name = {
    "jax_test": ag.Galaxies(galaxies=[galaxy]),
    "jax_test_group": ag.Galaxies(galaxies=[galaxy] + extra_galaxies),
}

for name, galaxies in galaxies_via_name.items():
    dataset_path = Path("dataset", "imaging", name)

    dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

    aplt.fits_imaging(
        dataset=dataset,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )

    ag.output_to_json(
        obj=galaxies,
        file_path=Path(dataset_path, "galaxies.json"),
    )

    print("Dataset written to", dataset_path)
