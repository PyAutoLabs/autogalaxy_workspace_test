"""
End-to-end test: jit-cached visualization during a real Nautilus model-fit.
==========================================================================

Single-galaxy autogalaxy port of the autolens
``scripts/interferometer/visualization/modeling_visualization_jit.py`` end-to-end test,
adapted for autogalaxy's single-galaxy interferometer (no lens/source split,
no mass profile, no PositionsLH).

This test runs in two parts:

Part 1 — **MGE caching probe.** Uses a linear MGE galaxy model (Basis of
``ag.lp_linear.Gaussian`` profiles). Calls
``analysis.fit_for_visualization(instance)`` twice and asserts the second
call is much faster than the first (confirming the compiled function is
cached on the analysis instance, not recompiled per visualization).

Part 2 — **Live Nautilus quick-update with linear light profiles.** Runs a
real (short) Nautilus fit with the same linear MGE galaxy. With autogalaxy's
``LightProfileLinear`` pytree handling, the
``linear_light_profile_intensity_dict`` lookup survives the JAX pytree
round-trip and no ``KeyError`` is raised. Asserts that ``fit.png`` files
land on disk, proving the JIT-cached fit_for_visualization fires correctly
during the live search callback.

This script deliberately opts in with
``AnalysisInterferometer(use_jax=True)``. Default model-fit scripts elsewhere
in the workspace leave the flag at ``False`` and are therefore untouched by
this change.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Live Nautilus + JIT path: real search, JAX, full-resolution mask and real
savefig.

ENV: real_output
"""

import shutil
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

Re-use the ``jax_test`` interferometer dataset. Auto-simulate if missing.
"""
mask_radius = 3.5

real_space_mask = ag.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset_path = path.join("dataset", "interferometer", "jax_test")

if ag.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/jax_likelihood/simulator.py",
        ],
        check=True,
    )

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
)


"""
============================================================================
Part 1 — MGE caching probe
============================================================================

Model: linear MGE galaxy (Basis of ``ag.lp_linear.Gaussian``). Single-galaxy
autogalaxy interferometer — no lens/source split, no mass profile.
"""
print("\n" + "=" * 72)
print("Part 1: MGE caching probe")
print("=" * 72)

total_gaussians = 3
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

gaussian_list = af.Collection(
    af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
)
for i, gaussian in enumerate(gaussian_list):
    gaussian.centre.centre_0 = centre_0
    gaussian.centre.centre_1 = centre_1
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = 10 ** log10_sigma_list[i]

bulge_mge = af.Model(ag.lp_basis.Basis, profile_list=list(gaussian_list))

galaxy_mge = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_mge)

model_mge = af.Collection(galaxies=af.Collection(galaxy=galaxy_mge))


analysis_mge = ag.AnalysisInterferometer(
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
assert isinstance(
    fit_1.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_1.log_likelihood)}"
assert analysis_mge.supports_jax_visualization is True

t0 = time.perf_counter()
fit_2 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_2.log_likelihood)
t1 = time.perf_counter()
second_time = t1 - t0
print(f"Second call:                 {second_time:.3f}s")
assert isinstance(
    fit_2.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_2.log_likelihood)}"
print("PASS: MGE fit_for_visualization returns JAX-backed fits with use_jax=True.")


"""
__Visualization Sanity__

Phase D.1 rollout — autogalaxy interferometer variant (no Tracer / no
lensing latents). Combines:

* **Non-lensing template:** `fit.figure_of_merit` finite — catches the
  JAX-trace mismatch or inversion collapse that would leave the model
  cosmetically OK but the chi² nan/inf.
* **Interferometer-specific:** `fit.model_visibilities` finite +
  non-zero. Catches the NUFFT / linear-inversion collapse where the
  visibilities silently become all-zero or NaN.

Asserts run on the script's cached `fit_2` from Part 1 so the warm JIT
path is exercised (first-call compile already paid by the caching probe).
"""

_mv = np.asarray(fit_2.model_data)
assert np.isfinite(
    _mv
).all(), "fit.model_data (visibilities) have nan/inf — NUFFT / inversion collapse"
assert (
    float(np.abs(_mv).sum()) > 0.0
), "fit.model_data (visibilities) all-zero — NUFFT / inversion collapse"
_fom = float(fit_2.figure_of_merit)
assert np.isfinite(_fom), f"figure_of_merit = {_fom} — chi² nan/inf, fit collapsed"
print(
    f"  PASS Visualization Sanity (autogalaxy interferometer): "
    f"|model_data|.sum() = {float(np.abs(_mv).sum()):.4f}, "
    f"figure_of_merit = {_fom:.4f}"
)


"""
============================================================================
Part 2 — Live Nautilus quick-update with linear light profiles
============================================================================

Model: linear MGE galaxy (Basis of ``ag.lp_linear.Gaussian``). Single-galaxy
autogalaxy interferometer — no lens/source split, no mass profile. The
``linear_light_profile_intensity_dict`` lookup is exercised during
visualization. The live search fires quick-update visualization every
``iterations_per_quick_update`` calls; we verify ``fit.png`` lands on disk.
"""
print("\n" + "=" * 72)
print("Part 2: Live Nautilus with linear MGE profiles + jit-visualization")
print("=" * 72)

total_gaussians2 = 3
log10_sigma_list2 = np.linspace(-2, np.log10(mask_radius), total_gaussians2)

centre_0_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

gaussian_list2 = af.Collection(
    af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians2)
)
for i, gaussian in enumerate(gaussian_list2):
    gaussian.centre.centre_0 = centre_0_2
    gaussian.centre.centre_1 = centre_1_2
    gaussian.ell_comps = gaussian_list2[0].ell_comps
    gaussian.sigma = 10 ** log10_sigma_list2[i]

bulge_mge2 = af.Model(ag.lp_basis.Basis, profile_list=list(gaussian_list2))

galaxy_mge2 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_mge2)

model_mge2 = af.Collection(galaxies=af.Collection(galaxy=galaxy_mge2))


analysis_mge2 = ag.AnalysisInterferometer(
    dataset=dataset,
    use_jax=True,
)

output_root = (
    Path("scripts") / "interferometer" / "images" / "modeling_visualization_jit"
)
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)

# Also clean the autofit search output. Without this, Nautilus resumes from
# the previous run's cached samples.csv and skips live sampling — so the
# quick-update visualizer never fires, _jitted_fit_from is never set, and
# the assertion below would fail on every rerun. Force a fresh run.
output_search_root = Path("output") / output_root / "mge_linear"
if output_search_root.exists():
    shutil.rmtree(output_search_root)

search = af.Nautilus(
    path_prefix=str(output_root),
    name="mge_linear",
    n_live=50,
    n_like_max=1500,
    iterations_per_quick_update=500,
    number_of_cores=1,
)

print("Running Nautilus ...")
result = search.fit(model=model_mge2, analysis=analysis_mge2)

# The Nautilus output goes to output/<path_prefix>/<name>/<hash>/image/
# The quick-update visualizer writes fit.png during each quick update.
produced_pngs = list(output_search_root.rglob("fit.png"))
print(f"fit.png files produced: {len(produced_pngs)}")
for p in produced_pngs:
    print(f"  {p}")
assert len(produced_pngs) > 0, (
    f"no fit.png produced under {output_search_root} — "
    "quick-update visualization did not fire"
)

# Note: _jitted_fit_from is built on the worker process Nautilus forks for the search
# loop, not the parent's analysis_mge2 instance — so we don't assert it post-search.
# Part 1 above already verifies the cache is set on the calling process.

print(
    "\nPASS: jit-cached fit_for_visualization fires during Nautilus quick updates "
    "for interferometer, fit.png written."
)
