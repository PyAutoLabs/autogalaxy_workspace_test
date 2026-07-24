"""
Visualization: Ellipse Analysis
================================

Exercises every public path through the ellipse visualization plotters
(``PlotterEllipse.imaging`` and ``PlotterEllipse.fit_ellipse``) end-to-end.

This is the regression target locked in before later prompts in the
``z_features/ellipse_fitting_jax.md`` feature rewrite the 300-iteration
mask-rejection loop in ``FitEllipse.points_from_major_axis_from`` and add
JAX paths underneath.  If any plotter regresses or a code path stops being
reachable, this script breaks loudly.

Dataset: reuses the ``jax_test`` single-galaxy imaging dataset from
``scripts/imaging/jax_likelihood/`` (auto-simulated on first run).
The PSF is ignored by ellipse fitting, so any imaging dataset works.

Scenarios
---------
Each scenario writes into its own subfolder under ``images/visualization/``:

    plain/        — single ``Ellipse`` with no perturbation
    multipole/    — ``Ellipse`` + ``EllipseMultipole(m=4)``
    scaled/       — ``Ellipse`` + ``EllipseMultipoleScaled(m=4)``
    masked/       — ``Ellipse`` over a small circular mask whose radius is
                    smaller than the ellipse's major axis, so the ellipse
                    perimeter crosses masked pixels and the 300-iteration
                    mask-rejection loop in
                    ``PyAutoGalaxy/autogalaxy/ellipse/fit_ellipse.py:81-134``
                    actually fires.

Each scenario calls ``VisualizerEllipse.visualize_before_fit`` and
``VisualizerEllipse.visualize`` and asserts the expected PNGs exist.
Numpy-only — no JAX imports.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Asserts subplot PNG / FITS land on disk (needs real plots) and reads full-
resolution data.

ENV: full_datasets real_plots
"""

import shutil
import subprocess
import sys
import time
from os import path
from pathlib import Path
from types import SimpleNamespace

from autogalaxy import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "..", "config"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "..", "images"),
)

import autofit as af
import autogalaxy as ag
from autogalaxy.ellipse.model.visualizer import VisualizerEllipse


"""
__Dataset__

Reuse the ``jax_test`` imaging dataset.  The simulator under
``scripts/imaging/jax_likelihood/`` writes the FITS files into
``dataset/imaging/jax_test/``; auto-simulate them on first run.
"""

dataset_path = path.join("dataset", "imaging", "jax_test")

if ag.util.dataset.should_simulate(dataset_path):
    subprocess.run(
        [sys.executable, "scripts/imaging/jax_likelihood/simulator.py"],
        check=True,
    )

dataset_unmasked = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)


"""
__Mask__

Two masks: a generous one for the first three scenarios (the ellipse sits
inside it cleanly) and a tight one for the masked scenario (perimeter
overlaps masked pixels).
"""

mask_generous = ag.Mask2D.circular(
    shape_native=dataset_unmasked.shape_native,
    pixel_scales=dataset_unmasked.pixel_scales,
    radius=3.0,
)
dataset_generous = dataset_unmasked.apply_mask(mask=mask_generous)

mask_tight = ag.Mask2D.circular(
    shape_native=dataset_unmasked.shape_native,
    pixel_scales=dataset_unmasked.pixel_scales,
    radius=0.95,
)
dataset_tight = dataset_unmasked.apply_mask(mask=mask_tight)


"""
__Models__

For each scenario, an ``af.Collection`` containing one ``Ellipse``, and
optionally a ``multipoles`` collection with a single ``EllipseMultipole`` or
``EllipseMultipoleScaled``.  All ellipse parameters are fixed so the
``instance_from_prior_medians()`` call below produces a deterministic instance.

The ellipse major-axis is chosen so the perimeter sits well inside
``mask_generous`` but crosses the boundary of ``mask_tight``, forcing the
300-iter loop to actually iterate for the masked scenario.
"""


def ellipse_model(major_axis: float):
    ellipse = af.Model(ag.Ellipse)
    ellipse.centre.centre_0 = 0.0
    ellipse.centre.centre_1 = 0.0
    ellipse.ell_comps.ell_comps_0 = 0.1
    ellipse.ell_comps.ell_comps_1 = 0.05
    ellipse.major_axis = major_axis
    return ellipse


def multipole_model(m: int, cos_amp: float, sin_amp: float):
    multipole = af.Model(ag.EllipseMultipole)
    multipole.m = m
    multipole.multipole_comps.multipole_comps_0 = cos_amp
    multipole.multipole_comps.multipole_comps_1 = sin_amp
    return multipole


def scaled_multipole_model(m: int, cos_amp: float, sin_amp: float, major_axis: float):
    multipole = af.Model(ag.EllipseMultipoleScaled)
    multipole.m = m
    multipole.scaled_multipole_comps.scaled_multipole_comps_0 = cos_amp
    multipole.scaled_multipole_comps.scaled_multipole_comps_1 = sin_amp
    multipole.major_axis = major_axis
    return multipole


major_axis = 1.0

model_plain = af.Collection(
    ellipses=af.Collection(ellipse_0=ellipse_model(major_axis=major_axis)),
)

model_multipole = af.Collection(
    ellipses=af.Collection(ellipse_0=ellipse_model(major_axis=major_axis)),
    multipoles=af.Collection(
        ellipse_0=af.Collection(m4=multipole_model(m=4, cos_amp=0.05, sin_amp=0.0)),
    ),
)

model_scaled = af.Collection(
    ellipses=af.Collection(ellipse_0=ellipse_model(major_axis=major_axis)),
    multipoles=af.Collection(
        ellipse_0=af.Collection(
            m4=scaled_multipole_model(
                m=4, cos_amp=0.05, sin_amp=0.0, major_axis=major_axis
            )
        ),
    ),
)


"""
__Scenarios__

Each scenario pairs a dataset (generous or tight mask) with a model.  The
"masked" scenario uses the tight mask to force the 300-iter loop to iterate.
"""

scenarios = [
    ("plain", dataset_generous, model_plain),
    ("multipole", dataset_generous, model_multipole),
    ("scaled", dataset_generous, model_scaled),
    ("masked", dataset_tight, model_plain),
]


"""
__Output Paths__

Clean the image root on each run so assertions reflect only this run's output.
"""

image_root = Path("scripts") / "ellipse" / "images" / "visualization"

if image_root.exists():
    shutil.rmtree(image_root)

image_root.mkdir(parents=True)


"""
__Run Each Scenario__

For each scenario:
  1. Build an ``AnalysisEllipse`` for the scenario's dataset.
  2. Build the ``ModelInstance`` via ``instance_from_prior_medians()``.
  3. Call ``VisualizerEllipse.visualize_before_fit`` and assert the dataset
     subplot was written.
  4. Call ``VisualizerEllipse.visualize`` and assert the ``fit_ellipse``
     plotter outputs (``ellipse_fit.png``, ``ellipse_residuals.png``,
     ``subplot_fit_ellipse.png``) were written.

The "masked" scenario additionally asserts the 300-iter loop is exercised by
inspecting that ``fit.points_from_major_axis_from()`` returns fewer points
than the unmasked ellipse would (i.e. the mask filter trimmed some points).
"""

expected_files = [
    "dataset.png",
    "dataset.fits",
    "ellipse_fit.png",
    "fit_ellipse.png",
    "ellipse_residuals.png",
]

for scenario_name, dataset, model in scenarios:
    print(f"\nRunning scenario: {scenario_name}")

    sub_path = image_root / scenario_name
    sub_path.mkdir(parents=True)
    sub_output = sub_path / "output"
    sub_output.mkdir(parents=True)
    sub_paths = SimpleNamespace(image_path=sub_path, output_path=sub_output)

    analysis = ag.AnalysisEllipse(
        dataset=dataset, title_prefix=scenario_name.upper(), use_jax=False
    )
    instance = model.instance_from_prior_medians()

    _t0 = time.perf_counter()
    VisualizerEllipse.visualize_before_fit(
        analysis=analysis, paths=sub_paths, model=model
    )
    VisualizerEllipse.visualize(
        analysis=analysis, paths=sub_paths, instance=instance, during_analysis=False
    )
    print(f"  visualize complete in {time.perf_counter() - _t0:.2f}s")

    for filename in expected_files:
        target = sub_path / filename
        assert target.exists(), f"{scenario_name}/{filename} missing"
        print(f"  {scenario_name}/{filename} OK")

    if scenario_name == "masked":
        fit_list = analysis.fit_list_from(instance=instance)
        ellipse = instance.ellipses[0]
        unmasked_total = ellipse.total_points_from(pixel_scale=dataset.pixel_scales[0])
        masked_total = fit_list[0].points_from_major_axis_from().shape[0]
        assert masked_total < unmasked_total, (
            f"masked scenario: expected the tight mask to drop ellipse points "
            f"(unmasked total = {unmasked_total}, after filter = {masked_total})"
        )
        print(
            f"  masked scenario trimmed ellipse points: "
            f"{unmasked_total} -> {masked_total}"
        )


print("\nAll ellipse visualization assertions passed.")
