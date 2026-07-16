# PyAutoGalaxy Workspace Test — Agent Instructions

This is the integration-test suite for **PyAutoGalaxy**, run on the build server to verify the core
library works end-to-end. It is **not** a user-facing workspace — see `../autogalaxy_workspace` for
examples and tutorials. These are the canonical, agent-agnostic instructions for this repo.

Dependencies: `autogalaxy`, `autofit`, `autoarray`, `numba`.

## Repository Structure

```
scripts/                     Integration-test scripts run on the build server
  imaging/ interferometer/   CCD imaging / interferometer model-fit tests
  ellipse/ quantity/         Ellipse-fitting and derived-quantity tests
  aggregator/                Results database aggregator tests (pytest-based)
  model_composition/         Model-composition tests
  jax_likelihood_functions/  JAX batched-likelihood tests
  jax_grad/ jax_assertions/  JAX gradient + assertion tests
  latent/                    Latent-variable tests
failed/                      One log per failing script
config/ output/              YAML config and runtime fit results
```

## Running Scripts

Run a single script directly from the repo root (with no env applied, searches run for real). The
aggregator tests are pytest-based and run from their directory:

```bash
python scripts/imaging/model_fit.py
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest scripts/aggregator -v
```

## Testing

On CI, `smoke_tests.yml` gates every PR on Python **3.12 and 3.13**. The gate runs the smoke runner
(the definition of green):

```bash
python .github/scripts/run_smoke.py
```

It executes the curated entries in `smoke_tests.txt`, applying per-entry environment from
`config/build/env_vars.yaml`. That file sets **fast-mode defaults for every entry** —
`PYAUTO_TEST_MODE=2` (skip the sampler), `PYAUTO_SMALL_DATASETS=1` (cap grids/masks),
`PYAUTO_FAST_PLOTS=1` — with **per-script `unset`/override** blocks where a test genuinely needs a
real sampler run or full-resolution data. So CI is *not* "searches run for real" by default; it is
fast-mode with targeted exceptions. A failure under these flags signals a real problem.

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/model_fit.py
```

## Aggregator & JAX Testing

- **Aggregator** (`scripts/aggregator/`) — `MockSearch` + `MockSamples` build mock model-fit
  results, scrape them into an SQLite database, then exercise each aggregator class (`GalaxiesAgg`,
  `FitImagingAgg`, `ImagingAgg`, `InterferometerAgg`, `EllipsesAgg`, `FitEllipseAgg`,
  `MultipolesAgg`, …) via its generator methods.
- **JAX** (`jax_likelihood_functions/`, `jax_grad/`, `jax_assertions/`) — exercises the `xp=jnp`
  path that library unit tests (NumPy-only) never touch, via `fitness._vmap`. See the PyAutoArray
  deep dive `../PyAutoArray/docs/agents/jax_and_decorators.md` for the boundary patterns.

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Related Repos

- Source libs: `../PyAutoGalaxy`, `../PyAutoArray`, `../PyAutoFit`, `../PyAutoConf`.
- `../autogalaxy_workspace` — the user-facing workspace; `../HowToGalaxy` — the tutorial series.
- `../PyAutoBuild` — CI / build tooling.
- `../autolens_assistant` — science-assistant workspace (literature wiki).

## Task Workflows

When a library change lands, run the smoke suite, read any `[FAIL]` entries, and update the affected
test scripts to the new API (preserving intent). **Never edit a script to mask a real regression** —
if a library bug surfaces, flag it for the source repo rather than papering over it. Note in your PR
any change that affects sibling repos (`autogalaxy_workspace`, the source libraries).

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
