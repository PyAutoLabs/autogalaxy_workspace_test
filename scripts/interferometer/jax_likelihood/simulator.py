"""
Simulator: JAX Interferometer Test Datasets
===========================================

Simulates the two `Interferometer` datasets consumed by every script in
``scripts/interferometer/jax_likelihood/`` (and the ``jax_grad`` /
``visualization`` scripts that re-use them).

A single galaxy with a Sersic bulge + Exponential disk is observed by a
synthetic interferometer with deterministic random uv-coverage. Fully
self-contained — no external uv-wavelength fixture is required, mirroring the
imaging port's all-inline-generation pattern.

**Two datasets, because the model in each script must match the data it fits.**
``mge_group.py`` fits a primary galaxy *plus* extra galaxies; fitting extra
galaxies against data that contains none is a model that does not match its
data. So this simulator writes:

- ``dataset/interferometer/jax_test`` — the single galaxy, fit by every other
  script.
- ``dataset/interferometer/jax_test_group`` — the same galaxy plus the two
  extra galaxies the ``mge_group.py`` model actually composes, at the same
  centres.

The real-space grid the visibilities are transformed from is set by what the
models resolve, not by what an array could deliver: 128 x 128 at 0.2"/pixel,
comfortably oversampling the 200-visibility uv-coverage below.

Output files (under each dataset directory):

- ``data.fits`` — simulated complex visibilities (real, imag stacked)
- ``noise_map.fits`` — per-visibility noise sigma
- ``uv_wavelengths.fits`` — the synthetic uv-coverage used by the simulator
- ``galaxies.json`` — the exact ``Galaxies`` used, for reproducibility

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

from pathlib import Path

import numpy as np

import autogalaxy as ag
import autogalaxy.plot as aplt


grid = ag.Grid2D.uniform(shape_native=(128, 128), pixel_scales=0.2)

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

extra_galaxy_centre_list = [(1.2, 1.2), (-1.0, 1.5)]

extra_galaxies = [
    ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.SersicSph(
            centre=centre,
            intensity=0.25,
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
    dataset_path = Path("dataset", "interferometer", name)
    dataset_path.mkdir(parents=True, exist_ok=True)

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
