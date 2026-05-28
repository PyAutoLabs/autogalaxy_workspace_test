"""
Visualization JAX Pilot: Imaging Analysis (autogalaxy)
=======================================================

Single-galaxy autogalaxy port of the autolens ``visualization_jax.py`` pilot
(https://github.com/PyAutoLabs/PyAutoFit/issues/1227).

Goal
----
Run ``VisualizerImaging.visualize`` with JAX enabled end-to-end via
``use_jax=True`` on ``Analysis``. After PyAutoGalaxy #390 (2026-05-08) the
imaging visualizer dispatches through ``analysis.fit_for_visualization``,
which lazily wraps ``fit_from`` in ``jax.jit``. Visualization now follows
``use_jax`` automatically. To trace across that boundary the model and fit
return type must be JAX pytrees, so this script enables pytree registration
before constructing the model. Parametric MGE galaxy — simplest case (no
pixelization, no inversion).

Scope
-----
- Parametric MGE galaxy only.
- Calls ``VisualizerImaging.visualize`` only (not ``visualize_before_fit``).
- Re-uses the ``jax_test`` dataset from ``jax_likelihood_functions/imaging``.
- Reuses ``config_source/visualize/plots.yaml`` from ``visualization.py`` so
  only ``fit.png`` and ``galaxies.png`` are attempted.
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

from autoconf import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config_source"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

import autofit as af
import autogalaxy as ag
from autogalaxy.imaging.model.visualizer import VisualizerImaging


"""
__Dataset__
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

if not path.exists(path.join(dataset_path, "data.fits")):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
        check=True,
    )

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

mask_radius = 3.0
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)


"""
__Model__

MGE parametric galaxy (matches the MGE pattern in
``jax_likelihood_functions/imaging/mge.py``).
"""
galaxy_bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=galaxy_bulge)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


"""
__Analysis__

``use_jax=True`` turns on the JAX ``_xp`` path. Visualization now follows
``use_jax`` automatically via the ``Analysis.fit_for_visualization`` helper.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
    title_prefix="JAX_PILOT",
)


"""
__Paths__
"""
image_path = Path("scripts") / "imaging" / "images" / "visualization_jax"
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)
output_path = image_path / "output"
output_path.mkdir(parents=True)
paths = SimpleNamespace(image_path=image_path, output_path=output_path)


"""
__Run visualize on the eager-JAX fit__
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerImaging.visualize with use_jax=True ...")
VisualizerImaging.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)
assert (image_path / "fit.png").exists(), "fit.png was not produced"
print("PILOT SUCCEEDED — JAX-backed visualization produced fit.png/galaxies.png.")


"""
__Visualization Sanity__

Phase D.2.a rollout — autogalaxy variant (no Tracer / no lensing
latents). Asserts `fit.model_data` finite + non-zero and
`fit.figure_of_merit` finite on the script's prior-median fit, catching
the analogous failure mode where the JAX path silently produces all-zero
or NaN model output despite the cosmetic plot looking fine.
"""
import numpy as _sanity_np

_fit_for_vis = analysis.fit_from(instance=instance)
_model_image = _sanity_np.asarray(_fit_for_vis.model_data)
assert _sanity_np.isfinite(
    _model_image
).all(), "fit.model_data has nan/inf — JAX-trace mismatch or inversion collapse"
assert (
    float(_sanity_np.abs(_model_image).sum()) > 0.0
), "fit.model_data all-zero — light profile evaluation collapsed"
_fom = float(_fit_for_vis.figure_of_merit)
assert _sanity_np.isfinite(
    _fom
), f"figure_of_merit = {_fom} — chi² nan/inf, fit collapsed"
print(
    f"  PASS Visualization Sanity (autogalaxy imaging): "
    f"|model_data|.sum() = {float(_sanity_np.abs(_model_image).sum()):.4f}, "
    f"figure_of_merit = {_fom:.4f}"
)
