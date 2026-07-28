"""
Integration guard: latent variables that go NaN in an arbitrary per-sample
pattern must NOT crash the end-of-search latent summary.

This is the autogalaxy analogue of the autolens / autofit
``latent_nan_robustness`` guards. It reproduces the same PyAutoFit bug
(``autofit/non_linear/analysis/analysis.py::compute_latent_samples``): the JAX
batch path masked finite latent *columns* **per batch**
(``jnp.all(isfinite, axis=0)``), so a single sample whose latent went NaN in
one batch dropped that latent's whole column for that batch only. Different
samples then carried different ``Sample.kwargs`` key sets, and
``Samples.summary()`` raised ``KeyError`` building its model from batch 0's
keys.

The ``PYAUTO_LATENT_NAN_INJECT=stride:N`` knob (``autonerves.test_mode``) sets
NaN on latent column 0 (``total_galaxy_0_flux``) for every sample whose
absolute index is a non-zero multiple of ``N``. With ``N >= batch_size`` batch 0
stays fully finite (seeds the model with both latent keys) and a later batch
loses column 0 — the inconsistency that crashes pre-fix.

The bug is JAX-only (the NumPy branch row-masks first and never produces
inconsistent keys), so the search runs on the NumPy path and latents are
materialised with a separate ``use_jax=True`` analysis. ``config/latent.yaml``
enables both ``total_galaxy_0_flux`` and ``total_galaxy_0_flux_mujy`` so the
multi-column mis-zip path is exercised; ``magzero`` is supplied so the µJy
latent is finite.

PASS (post-fix): ``summary()`` / ``median_pdf()`` succeed and surviving latents
are finite. FAIL (pre-fix): ``KeyError``. Structural regression guard, not a
Bayesian validation.
"""

import math
import os

os.environ["PYAUTO_LATENT_NAN_INJECT"] = "stride:3"
# Skip the search's incidental post-fit latent pass (the NumPy path, not the
# branch under test). The explicit compute_latent_samples call below is NOT
# gated by this flag, so the JAX path we care about still runs.
os.environ["PYAUTO_SKIP_LATENTS"] = "1"

import autofit as af
import autogalaxy as ag
from autogalaxy import fixtures
from autogalaxy.imaging.model.latent import LATENT_FUNCTIONS

LATENT_BATCH_SIZE = 3  # <= the stride above, so batch 0 stays fully finite.

dataset = fixtures.make_masked_imaging_7x7()

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic)
model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

# Search on the NumPy path; the latent masking bug is JAX-only.
analysis = ag.AnalysisImaging(dataset=dataset, use_jax=False, magzero=25.0)

search = af.Nautilus(
    name="latent_nan_robustness",
    n_live=15,
    n_like_max=30,
)

result = search.fit(model=model, analysis=analysis)

assert len(result.samples.sample_list) > LATENT_BATCH_SIZE, (
    f"Need >{LATENT_BATCH_SIZE} samples for a multi-batch latent run; got "
    f"{len(result.samples.sample_list)}. Increase n_live / n_like_max."
)

analysis_jax = ag.AnalysisImaging(dataset=dataset, use_jax=True, magzero=25.0)

latent_samples = analysis_jax.compute_latent_samples(
    result.samples, batch_size=LATENT_BATCH_SIZE
)

assert latent_samples is not None, (
    "compute_latent_samples returned None — expected a populated latent Samples "
    "object (check config/latent.yaml enables the flux latents)."
)

# These two calls crash pre-fix (KeyError in parameter_lists_for_paths).
summary = latent_samples.summary()
instance = latent_samples.median_pdf()

surviving = [k for k in LATENT_FUNCTIONS if hasattr(instance, k)]
assert (
    surviving
), "No latents survived — expected at least total_galaxy_0_flux variants."

for key in surviving:
    value = float(getattr(instance, key))
    assert math.isfinite(
        value
    ), f"Surviving latent '{key}' is not finite (got {value})."

print(
    f"PASSED: latent summary survived arbitrary NaN injection "
    f"({len(surviving)} latents finite, batch_size={LATENT_BATCH_SIZE})."
)
for key in surviving:
    print(f"  {key}: {float(getattr(instance, key)):.6g}")
