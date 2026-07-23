"""
Visualization JAX Pilot: Ellipse Analysis (autogalaxy)
======================================================

Tests that ``VisualizerEllipse.visualize`` remains compatible with
``AnalysisEllipse(use_jax=True)``. The ellipse visualizer intentionally builds
NumPy-backed fit lists for plotting, while the analysis itself still advertises
JAX-capable visualization support.

Scope
-----
- Single ``af.Model(ag.Ellipse)`` model — no multipoles. Multipoles are
  exercised by the non-JAX ``ellipse/visualization.py`` already.
- Calls ``VisualizerEllipse.visualize`` only (not ``visualize_before_fit``).
- Reuses the ``dataset/imaging/jax_test`` dataset that the
  ``jax_likelihood_functions`` scripts produce.
- ``use_jax=True`` turns on the JAX-capable analysis path.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX-capable ellipse visualization path; needs JAX enabled, real plots and
full-resolution data.

ENV: jax full_datasets real_plots
"""

import shutil
import subprocess
import sys
from os import path
from pathlib import Path
from types import SimpleNamespace

from autogalaxy import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

import autofit as af
import autogalaxy as ag
from autogalaxy.ellipse.model.visualizer import VisualizerEllipse


"""
__Dataset__

Reuse the ``jax_test`` imaging dataset (auto-simulated on first run).
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

if not path.exists(path.join(dataset_path, "data.fits")):
    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
        check=True,
    )

dataset_unmasked = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

mask = ag.Mask2D.circular(
    shape_native=dataset_unmasked.shape_native,
    pixel_scales=dataset_unmasked.pixel_scales,
    radius=3.0,
)
dataset = dataset_unmasked.apply_mask(mask=mask)


"""
__Model__

A single fixed ``Ellipse`` so ``instance_from_prior_medians()`` produces
a deterministic instance with the major-axis well inside the mask.
"""
ellipse = af.Model(ag.Ellipse)
ellipse.centre.centre_0 = 0.0
ellipse.centre.centre_1 = 0.0
ellipse.ell_comps.ell_comps_0 = 0.1
ellipse.ell_comps.ell_comps_1 = 0.05
ellipse.major_axis = 1.0

model = af.Collection(ellipses=af.Collection(ellipse_0=ellipse))


"""
__Analysis__

``use_jax=True`` turns on the JAX-capable analysis path. Ellipse plotting still
uses NumPy-backed fit lists because matplotlib is the consumer.
"""
analysis = ag.AnalysisEllipse(
    dataset=dataset,
    use_jax=True,
    title_prefix="JAX_PILOT",
)
assert analysis.supports_jax_visualization is True


"""
__Paths__
"""
image_path = Path("scripts") / "ellipse" / "images" / "visualization_jax"
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

print("Running VisualizerEllipse.visualize with use_jax=True ...")
VisualizerEllipse.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)

# `fit_ellipse.png` is one of the artifacts that the non-JAX
# `ellipse/visualization.py` script asserts on; check the same one here.
assert (
    image_path / "fit_ellipse.png"
).exists(), "fit_ellipse.png was not produced by the JAX-backed visualizer"
print("PILOT SUCCEEDED — JAX-backed ellipse visualization produced fit_ellipse.png.")


"""
__Visualization Sanity__

Phase D.2.b.ii — autogalaxy ellipse variant (no Tracer / no lensing
latents). ``FitEllipseSummed`` is the fit object; ``figure_of_merit`` is
the aggregate log-likelihood the search would use. Asserts FoM is finite
on the prior-median instance, catching JAX-trace mismatches that would
leave the cosmetic plot OK but the underlying log-likelihood nan/inf.

Note: ``FitEllipseSummed`` doesn't expose a single ``model_data`` array
the way ``FitImaging`` does — the model field lives per-ellipse on the
component ``FitEllipse`` objects. The figure_of_merit check covers the
end-to-end fit; per-component model_data assertion would just duplicate
the FoM signal with extra fragility.
"""
import numpy as _sanity_np

_fit_for_vis = analysis.fit_from(instance=instance)
_fom = float(_fit_for_vis.figure_of_merit)
assert _sanity_np.isfinite(
    _fom
), f"figure_of_merit = {_fom} — chi² nan/inf, fit collapsed"
print(
    f"  PASS Visualization Sanity (autogalaxy ellipse): "
    f"figure_of_merit = {_fom:.4f}"
)
