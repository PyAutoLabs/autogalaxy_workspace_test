"""
Visualization: Interferometer Analysis (autogalaxy)
====================================================

Tests that ``VisualizerInterferometer.visualize`` outputs expected files to disk
for an autogalaxy single-galaxy interferometer analysis (no lens/source split,
no mass profile).

Dataset: single-galaxy ``jax_test`` interferometer (MGE bulge simulator).

Structure
---------
Calls ``VisualizerInterferometer.visualize`` once with a parametric MGE galaxy
into a dedicated subfolder, asserting that ``fit.png`` is produced.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Asserts subplot PNG / FITS land on disk (needs real plots) and reads full-
resolution data.

ENV: full_datasets real_plots
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

import autofit as af
import autogalaxy as ag
from autogalaxy.interferometer.model.visualizer import VisualizerInterferometer


"""
__Dataset__

Reuse the ``jax_test`` interferometer dataset from
``scripts/jax_likelihood_functions/interferometer``.
Auto-simulate if the dataset is missing.
"""
dataset_path = path.join("dataset", "interferometer", "jax_test")

if not path.exists(path.join(dataset_path, "data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/jax_likelihood_functions/interferometer/simulator.py",
        ],
        check=True,
    )

mask_radius = 3.0

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)


"""
__Model__

Single-galaxy MGE parametric model — mirrors the canonical autogalaxy
interferometer pattern from ``jax_likelihood_functions/interferometer/mge.py``.
"""
bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


"""
__Analysis__

Explicit NumPy path (``use_jax=False``) — this is the NumPy baseline script.
"""
analysis = ag.AnalysisInterferometer(dataset=dataset, use_jax=False)


"""
__Paths__

Minimal paths stub: ``VisualizerInterferometer`` only needs ``image_path`` and
``output_path``. Clean the output directory on each run so assertions reflect
this run only.
"""
image_path = Path("scripts") / "interferometer" / "images" / "visualization"

if image_path.exists():
    shutil.rmtree(image_path)

image_path.mkdir(parents=True)

output_path = image_path / "output"
output_path.mkdir(parents=True)

paths = SimpleNamespace(
    image_path=image_path,
    output_path=output_path,
)


"""
__Visualize__

Run ``VisualizerInterferometer.visualize`` once with the MGE galaxy instance
and assert that ``fit.png`` lands on disk.
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerInterferometer.visualize (NumPy) ...")
VisualizerInterferometer.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)

print(list(image_path.iterdir()))
assert (image_path / "fit.png").exists(), "fit.png missing"
print("NumPy interferometer visualization produced fit.png.")
