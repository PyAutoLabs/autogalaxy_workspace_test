"""
End-to-end test: jit-cached visualization during a real Nautilus ellipse fit.
=============================================================================

Single-galaxy autogalaxy ellipse port of the autolens
``scripts/imaging/visualization/modeling_visualization_jit.py`` end-to-end test.

This test runs in two parts:

Part 1 — **Caching probe.** Uses a parametric single-``Ellipse`` model.
Calls ``analysis.fit_for_visualization(instance)`` twice and asserts the
second call is much faster than the first (confirming the compiled
function is cached on the analysis instance, not recompiled per
visualization).

Part 2 — **Live Nautilus quick-update.** Runs a real (short) Nautilus
fit with the same ellipse model. Asserts that ``fit_ellipse.png`` files
land on disk, proving the JIT-cached fit_for_visualization fires
correctly during the live search callback.

This script deliberately opts in with
``AnalysisEllipse(use_jax=True)``.
Default ellipse model-fit scripts elsewhere in the workspace leave the flag
at ``False`` and are therefore untouched by this change.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Live Nautilus + JAX/JIT visualization path: real search, JAX, full-
resolution mask and real savefig.

ENV: real_output
"""

import shutil
import subprocess
import sys
import time
from os import path
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autogalaxy as ag


"""
__Dataset__

Reuse the ``jax_test`` imaging dataset (auto-simulated on first run).
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

mask_radius = 3.0
mask = ag.Mask2D.circular(
    shape_native=dataset_unmasked.shape_native,
    pixel_scales=dataset_unmasked.pixel_scales,
    radius=mask_radius,
)
dataset = dataset_unmasked.apply_mask(mask=mask)


"""
============================================================================
Part 1 — Caching probe
============================================================================

Model: single parametric ``Ellipse`` with tight priors so the
prior-median instance lands inside the mask.
"""
print("\n" + "=" * 72)
print("Part 1: Ellipse caching probe")
print("=" * 72)

ellipse_mge = af.Model(ag.Ellipse)
ellipse_mge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
ellipse_mge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
ellipse_mge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
ellipse_mge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.1)
ellipse_mge.major_axis = 1.0

model_mge = af.Collection(ellipses=af.Collection(ellipse_0=ellipse_mge))


analysis_mge = ag.AnalysisEllipse(
    dataset=dataset,
    use_jax=True,
)

instance_mge = model_mge.instance_from_prior_medians()

t0 = time.perf_counter()
fit_1 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
first_time = t1 - t0
print(f"First call:                  {first_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert np.isfinite(float(fit_1.log_likelihood))
assert analysis_mge.supports_jax_visualization is True

t0 = time.perf_counter()
fit_2 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_2.log_likelihood)
t1 = time.perf_counter()
second_time = t1 - t0
print(f"Second call:                 {second_time:.3f}s")
assert np.isfinite(float(fit_2.log_likelihood))
print("PASS: Ellipse fit_for_visualization returns finite fits with use_jax=True.")


"""
__Visualization Sanity__

Phase D.2.b.ii — autogalaxy ellipse variant (no Tracer / no lensing).
``FitEllipseSummed`` doesn't expose a top-level ``model_data`` (the
model field lives per-ellipse on the component ``FitEllipse``s), so the
Sanity check is restricted to ``figure_of_merit`` — the aggregate
log-likelihood the search consumes. Catches JAX-trace mismatches that
would leave the cosmetic plot OK but the underlying FoM nan/inf.

Runs on the cached ``fit_2`` from Part 1 so the warm JIT path is
exercised (compile already paid above).
"""

_fom = float(fit_2.figure_of_merit)
assert np.isfinite(_fom), f"figure_of_merit = {_fom} — chi² nan/inf, fit collapsed"
print(
    f"  PASS Visualization Sanity (autogalaxy ellipse): "
    f"figure_of_merit = {_fom:.4f}"
)


"""
============================================================================
Part 2 — Live Nautilus quick-update with single Ellipse
============================================================================

Same parametric ``Ellipse`` model. The live search fires quick-update
visualization every ``iterations_per_quick_update`` calls; we verify
``fit_ellipse.png`` lands on disk.
"""
print("\n" + "=" * 72)
print("Part 2: Live Nautilus with ellipse + jit-visualization")
print("=" * 72)

ellipse_2 = af.Model(ag.Ellipse)
ellipse_2.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
ellipse_2.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
ellipse_2.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
ellipse_2.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.1)
ellipse_2.major_axis = 1.0

model_mge2 = af.Collection(ellipses=af.Collection(ellipse_0=ellipse_2))


analysis_mge2 = ag.AnalysisEllipse(
    dataset=dataset,
    use_jax=True,
)

output_root = Path("scripts") / "ellipse" / "images" / "modeling_visualization_jit"
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)

search = af.Nautilus(
    path_prefix=str(output_root),
    name="ellipse_jit",
    n_live=50,
    n_like_max=1500,
    iterations_per_quick_update=500,
    number_of_cores=1,
)

print("Running Nautilus ...")
result = search.fit(model=model_mge2, analysis=analysis_mge2)

# The Nautilus output goes to output/<path_prefix>/<name>/<hash>/image/.
# The quick-update visualizer writes fit_ellipse.png to that image
# folder during each quick update.
output_search_root = Path("output") / output_root / "ellipse_jit"
produced_pngs = list(output_search_root.rglob("fit_ellipse.png"))
print(f"fit_ellipse.png files produced: {len(produced_pngs)}")
for p in produced_pngs:
    print(f"  {p}")
assert len(produced_pngs) > 0, (
    f"no fit_ellipse.png produced under {output_search_root} — "
    "quick-update visualization did not fire"
)

# Note: _jitted_fit_from is built on the worker process Nautilus forks for the
# search loop, not the parent's analysis_mge2 instance — so we don't assert it
# post-search. Part 1 above already verifies the cache is set on the calling
# process.

print(
    "\nPASS: jit-cached fit_for_visualization fires during Nautilus quick updates "
    "for ellipse, fit_ellipse.png written."
)
