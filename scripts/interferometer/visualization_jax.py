"""
Visualization JAX Pilot: Interferometer Analysis (autogalaxy)
=============================================================

Single-galaxy autogalaxy port of the autolens ``visualization_jax.py`` pilot
for interferometer data.

Goal
----
Run ``VisualizerInterferometer.visualize`` with JAX enabled end-to-end, gated
behind ``use_jax_for_visualization=True`` on ``AnalysisInterferometer``. The
interferometer visualizer dispatches through ``analysis.fit_for_visualization``,
which lazily wraps ``fit_from`` in ``jax.jit``. To trace across that boundary
the model and fit return type must be JAX pytrees, so this script enables
pytree registration before constructing the model.

Scope
-----
- Parametric MGE galaxy only (no pixelization, no inversion).
- Calls ``VisualizerInterferometer.visualize`` only (not ``visualize_before_fit``).
- Re-uses the ``jax_test`` interferometer dataset from
  ``jax_likelihood_functions/interferometer``.
- Single-galaxy autogalaxy model — no lens/source split, no mass profile.
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

import autofit as af
import autogalaxy as ag
from autofit.jax.pytrees import enable_pytrees, register_model
from autogalaxy.interferometer.model.visualizer import VisualizerInterferometer

enable_pytrees()


"""
__Dataset__

Re-use the ``jax_test`` interferometer dataset. Auto-simulate if missing.
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

Single-galaxy MGE parametric model. Pytree registration is required before
constructing the model so that the model and fit return type are valid JAX
pytrees at the ``jax.jit`` boundary.
"""
bulge = ag.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

register_model(model)


"""
__Analysis__

``use_jax=True`` turns on the JAX ``_xp`` path; ``use_jax_for_visualization=True``
tells the search-level visualization path to wrap ``fit_from`` in ``jax.jit``
via the ``Analysis.fit_for_visualization`` helper.
"""
analysis = ag.AnalysisInterferometer(
    dataset=dataset,
    use_jax=True,
    use_jax_for_visualization=True,
    title_prefix="JAX_PILOT",
)


"""
__Paths__
"""
image_path = Path("scripts") / "interferometer" / "images" / "visualization_jax"
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

print("Running VisualizerInterferometer.visualize with use_jax_for_visualization=True ...")
VisualizerInterferometer.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)
assert (image_path / "fit.png").exists(), "fit.png was not produced"
print("PILOT SUCCEEDED — JAX-backed interferometer visualization produced fit.png.")
