"""
Simulator: JAX Interferometer Test Dataset
==========================================

Simulates the `Interferometer` dataset consumed by every script in
``scripts/jax_likelihood_functions/interferometer/``.

A single galaxy with a Sersic bulge + Exponential disk is observed by a
synthetic interferometer with deterministic random uv-coverage. Fully
self-contained — no external uv-wavelength fixture is required, mirroring the
imaging port's all-inline-generation pattern.

Output files (under ``dataset/interferometer/jax_test/``):

- ``data.fits`` — simulated complex visibilities (real, imag stacked)
- ``noise_map.fits`` — per-visibility noise sigma
- ``uv_wavelengths.fits`` — the synthetic uv-coverage used by the simulator
- ``galaxies.json`` — the exact ``Galaxies`` used, for reproducibility
"""

from pathlib import Path

import numpy as np

import autogalaxy as ag
import autogalaxy.plot as aplt


dataset_path = Path("dataset", "interferometer", "jax_test")
dataset_path.mkdir(parents=True, exist_ok=True)

grid = ag.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.1)

rng = np.random.default_rng(seed=1)
n_visibilities = 200
uv_wavelengths = rng.uniform(low=-1.0e5, high=1.0e5, size=(n_visibilities, 2))

simulator = ag.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=ag.TransformerDFT,
    noise_seed=1,
)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
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

galaxies = ag.Galaxies(galaxies=[galaxy])

dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

aplt.fits_interferometer(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

ag.output_to_json(
    obj=galaxies,
    file_path=Path(dataset_path, "galaxies.json"),
)

print("Dataset written to", dataset_path)
