# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**autogalaxy_workspace_test** is the integration test suite for PyAutoGalaxy. It contains pytest test scripts that are run on the build server to verify that the core PyAutoGalaxy functionality works end-to-end. It is not a user-facing workspace — see `../autogalaxy_workspace` for example scripts and tutorials.

Dependencies: `autogalaxy`, `autofit`, `numba`. Python version: 3.12.

## Workspace Structure

```
scripts/                     Integration test scripts run on the build server
  aggregator/                Database aggregator tests (pytest-based)
    conftest.py              Shared fixtures: model, samples, aggregator_from helper
    config/                  Local config for aggregator tests
    test_aggregator_*.py     Test files for each aggregator module
failed/                      Failure logs written here when a script errors
config/                      Root-level YAML configuration files
output/                      Model-fit results written here at runtime
```

## Running Tests

Aggregator tests use pytest and run from their directory:

```bash
cd scripts/aggregator
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest . -v
```

To run all scripts via the runner:

```bash
bash run_all_scripts.sh
```

## Aggregator Tests

Seven test files covering all PyAutoGalaxy aggregator modules:

| File | Aggregator Classes Tested |
|------|--------------------------|
| `test_aggregator_galaxies.py` | `GalaxiesAgg` |
| `test_aggregator_fit_imaging.py` | `FitImagingAgg` |
| `test_aggregator_imaging.py` | `ImagingAgg` |
| `test_aggregator_fit_interferometer.py` | `FitInterferometerAgg` |
| `test_aggregator_interferometer.py` | `InterferometerAgg` |
| `test_aggregator_ellipses.py` | `EllipsesAgg` |
| `test_aggregator_fit_ellipse.py` | `FitEllipseAgg` |
| `test_aggregator_multipoles.py` | `MultipolesAgg` |

Tests use `MockSearch` + `MockSamples` to create mock model-fit results, scrape them into an SQLite database, then test the aggregator's generator methods.

## Line Endings — Always Unix (LF)

All files **must use Unix line endings (LF, `\n`)**. Never write `\r\n` line endings.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
