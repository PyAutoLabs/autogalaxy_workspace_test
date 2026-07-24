"""
Simulator: JAX Ellipse Test Dataset
====================================

Simulates a small `Imaging` dataset consumed by every script in
``scripts/ellipse/jax_likelihood/``.

A single galaxy with a Sersic bulge is imaged at moderate signal-to-noise.
No lens / mass / source plane — this is a single-plane autogalaxy dataset
designed to exercise the JAX ellipse-fitting likelihood path.

Note: ellipse fitting does not use the PSF during the fit, but the PSF is
written to ``psf.fits`` to maintain ``Imaging.from_fits`` symmetry with the
other ``jax_likelihood`` simulators.

Output files (under ``dataset/ellipse/jax_test/``):

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


dataset_path = Path("dataset", "ellipse", "jax_test")

grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

psf = ag.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales, normalize=True
)

simulator = ag.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=1.0,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
        intensity=4.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

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
