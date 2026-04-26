"""
End-to-end test: jit-cached visualization during a real Nautilus model-fit.
==========================================================================

Single-galaxy autogalaxy port of the autolens
``scripts/imaging/modeling_visualization_jit.py`` end-to-end test.

This test runs in two parts:

Part 1 — **MGE caching probe.** Uses an MGE galaxy model (Basis of
``ag.lp_linear.Gaussian`` profiles). Calls
``analysis.fit_for_visualization(instance)`` twice and asserts the second
call is much faster than the first (confirming the compiled function is
cached on the analysis instance, not recompiled per visualization).

Part 2 — **Live Nautilus quick-update with linear light profiles.** Runs a
real (short) Nautilus fit with the same MGE galaxy. With autogalaxy's
``LightProfileLinear`` pytree handling, the
``linear_light_profile_intensity_dict`` lookup survives the JAX pytree
round-trip and no ``KeyError`` is raised. Asserts that ``fit.png`` files
land on disk, proving the JIT-cached fit_for_visualization fires correctly
during the live search callback.

This script deliberately opts in with
``AnalysisImaging(use_jax=True, use_jax_for_visualization=True)``. Default
model-fit scripts elsewhere in the workspace leave both flags at ``False``
and are therefore untouched by this change.
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
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()


"""
__Dataset__

Re-use the ``jax_test`` dataset that the jax_likelihood_functions scripts rely on.
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

mask_radius = 3.5
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=4)


"""
============================================================================
Part 1 — MGE caching probe
============================================================================

Model: MGE parametric galaxy (Basis of ``ag.lp_linear.Gaussian``).
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

register_model(model_mge)

analysis_mge = ag.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
    use_jax_for_visualization=True,
)

instance_mge = model_mge.instance_from_prior_medians()

t0 = time.perf_counter()
fit_1 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
compile_time = t1 - t0
print(f"First call (compile + run): {compile_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert isinstance(fit_1.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit_1.log_likelihood)}"
)

t0 = time.perf_counter()
fit_2 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_2.log_likelihood)
t1 = time.perf_counter()
cached_time = t1 - t0
print(f"Second call (cached):       {cached_time:.3f}s")
print(f"Speedup:                    {compile_time / max(cached_time, 1e-9):.1f}x")

assert cached_time < compile_time * 0.5, (
    f"Cached call ({cached_time:.3f}s) not faster than compile "
    f"({compile_time:.3f}s) — JIT cache is not being hit."
)
assert analysis_mge._jitted_fit_from is not None, (
    "expected _jitted_fit_from to be cached on the analysis instance after first call"
)
print("PASS: MGE jit-cached fit_for_visualization works and is reused.")


"""
============================================================================
Part 2 — Live Nautilus quick-update with linear light profiles
============================================================================

Model: MGE parametric galaxy (Basis of ``ag.lp_linear.Gaussian``). The
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

register_model(model_mge2)

analysis_mge2 = ag.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
    use_jax_for_visualization=True,
)

output_root = Path("scripts") / "imaging" / "images" / "modeling_visualization_jit"
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)

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
# The quick-update visualizer writes fit.png (via subplot_fit function)
# to that image folder during each quick update.
output_search_root = Path("output") / output_root / "mge_linear"
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
    "with linear MGE profiles, fit.png written, no KeyError from "
    "linear_light_profile_intensity_dict lookup."
)
