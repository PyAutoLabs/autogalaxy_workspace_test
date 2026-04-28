"""
Simulator: JAX Multi-Wavelength Test Dataset
============================================

Simulates two-band (g and r) ``Imaging`` datasets consumed by every script in
``scripts/jax_likelihood_functions/multi/``.

A single galaxy with a Sersic bulge + Exponential disk is observed in two
wavebands. Each band has a different bulge intensity to give chromatic
variation, and a distinct noise seed. No lens / mass / source plane — this
is single-plane autogalaxy data designed to exercise the JAX likelihood path
through ``af.FactorGraphModel`` over multiple datasets.

Output files (under ``dataset/multi/jax_test/``):

- ``{g,r}_data.fits`` — simulated noisy images
- ``{g,r}_psf.fits`` — Gaussian PSF kernels
- ``{g,r}_noise_map.fits`` — per-pixel 1-sigma noise maps
- ``galaxies.json`` — the g-band ``Galaxies`` (the r-band variant differs only
  by ``bulge.intensity``), for reproducibility
"""

from pathlib import Path

import autogalaxy as ag


dataset_path = Path("dataset", "multi", "jax_test")
dataset_path.mkdir(parents=True, exist_ok=True)

grid = ag.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.1)

psf = ag.Convolver.from_gaussian(
    shape_native=(21, 21), sigma=0.1, pixel_scales=grid.pixel_scales, normalize=True
)


def _galaxies_for_band(bulge_intensity: float) -> ag.Galaxies:
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
    return ag.Galaxies(galaxies=[galaxy])


# g-band: bulge intensity 1.0; r-band: bulge intensity 1.6 — chromatic variation.
band_intensity = {"g": 1.0, "r": 1.6}
band_seed = {"g": 1, "r": 2}

for band, intensity in band_intensity.items():
    simulator = ag.SimulatorImaging(
        exposure_time=2000.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=band_seed[band],
    )
    galaxies = _galaxies_for_band(bulge_intensity=intensity)
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
    print(f"Saved {band}-band dataset")

print("Multi-wavelength datasets written to", dataset_path)
