"""
Simulator: JAX Multi-Wavelength Test Datasets
=============================================

Simulates the two-band (g and r) ``Imaging`` datasets consumed by every script
in ``scripts/multi_dataset/jax_likelihood/`` (and the ``jax_grad`` scripts that
re-use them).

A single galaxy with a Sersic bulge + Exponential disk is observed in two
wavebands. Each band has a different bulge intensity to give chromatic
variation, and a distinct noise seed. No lens / mass / source plane — this
is single-plane autogalaxy data designed to exercise the JAX likelihood path
through ``af.FactorGraphModel`` over multiple datasets.

**Two datasets, because the model in each script must match the data it fits.**
``mge_group.py`` fits a primary galaxy *plus* extra galaxies; fitting extra
galaxies against data that contains none is a model that does not match its
data. So this simulator writes:

- ``dataset/multi_dataset/jax_test`` — the single galaxy, fit by every other
  script.
- ``dataset/multi_dataset/jax_test_group`` — the same galaxy plus the two extra
  galaxies the ``mge_group.py`` model actually composes, at the same centres.

Resolution is set by what the models resolve: an 80 x 80 grid at 0.2"/pixel
with a 0.25" Gaussian PSF sigma, which stays well sampled at 0.2"/pixel.

Output files (under each dataset directory):

- ``{g,r}_data.fits`` — simulated noisy images
- ``{g,r}_psf.fits`` — Gaussian PSF kernels
- ``{g,r}_noise_map.fits`` — per-pixel 1-sigma noise maps
- ``galaxies.json`` — the g-band ``Galaxies`` (the r-band variant differs only
  by the bulge intensities), for reproducibility

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

from pathlib import Path

import autogalaxy as ag


grid = ag.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.2)

psf = ag.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.25, pixel_scales=grid.pixel_scales, normalize=True
)

extra_galaxy_centre_list = [(1.2, 1.2), (-1.0, 1.5)]


def _galaxies_for_band(
    bulge_intensity: float,
    extra_intensity: float,
    with_extra_galaxies: bool,
) -> ag.Galaxies:
    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=bulge_intensity,
            effective_radius=0.6,
            sersic_index=3.0,
        ),
        disk=ag.lp.Exponential(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
            intensity=0.5,
            effective_radius=1.6,
        ),
    )

    galaxy_list = [galaxy]

    if with_extra_galaxies:
        galaxy_list += [
            ag.Galaxy(
                redshift=0.5,
                bulge=ag.lp.SersicSph(
                    centre=centre,
                    intensity=extra_intensity,
                    effective_radius=0.35,
                    sersic_index=2.0,
                ),
            )
            for centre in extra_galaxy_centre_list
        ]

    return ag.Galaxies(galaxies=galaxy_list)


# g-band: bulge intensity 1.0; r-band: bulge intensity 1.6 — chromatic variation.
band_intensity = {"g": 1.0, "r": 1.6}
# The extra galaxies are chromatic too: fainter in g than in r.
band_extra_intensity = {"g": 0.25, "r": 0.4}
band_seed = {"g": 1, "r": 2}

for name, with_extra_galaxies in (("jax_test", False), ("jax_test_group", True)):
    dataset_path = Path("dataset", "multi_dataset", name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    for band, intensity in band_intensity.items():
        simulator = ag.SimulatorImaging(
            exposure_time=2000.0,
            psf=psf,
            background_sky_level=0.1,
            add_poisson_noise_to_data=True,
            noise_seed=band_seed[band],
        )
        galaxies = _galaxies_for_band(
            bulge_intensity=intensity,
            extra_intensity=band_extra_intensity[band],
            with_extra_galaxies=with_extra_galaxies,
        )
        dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

        ag.output_to_fits(
            values=dataset.data.native,
            file_path=dataset_path / f"{band}_data.fits",
            overwrite=True,
        )
        ag.output_to_fits(
            values=dataset.psf.kernel.native,
            file_path=dataset_path / f"{band}_psf.fits",
            overwrite=True,
        )
        ag.output_to_fits(
            values=dataset.noise_map.native,
            file_path=dataset_path / f"{band}_noise_map.fits",
            overwrite=True,
        )
        if band == "g":
            ag.output_to_json(
                obj=galaxies,
                file_path=dataset_path / "galaxies.json",
            )
        print(f"Saved {band}-band dataset to {dataset_path}")

print("Multi-wavelength datasets written")
