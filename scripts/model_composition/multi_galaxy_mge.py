"""
Model Composition: Multi-Galaxy MGE Integration Tests
======================================================

Integration tests for realistic multi-galaxy model composition using
PyAutoGalaxy. These tests compose models at the same complexity as actual
science runs — multiple galaxies sharing one plane, each with a Multi-Gaussian
Expansion (MGE) light basis — and assert structural properties of the resulting
model graph.

This catches silent PyAutoFit regressions that alter how autogalaxy models are
composed. A particularly dangerous class of bug is when a refactor changes
which priors are shared by object identity (e.g. moving a prior construction
across a loop boundary), producing a model with the correct type and component
count but the wrong degrees of freedom.

The ``identifier`` regression anchor is the most important assertion: if a
PyAutoFit refactor changes the identifier for an unchanged model, it means
existing users' output folders will no longer match, breaking backwards
compatibility.

__Contents__

- **Galaxy 0 Composition:** MGE bulge with gaussian_per_basis=2.
- **Galaxy 1 Composition:** MGE bulge with gaussian_per_basis=1.
- **Full Model:** af.Collection wrapping galaxy_0 + galaxy_1 (both at the same redshift).
- **MGE Prior Identity:** Within-basis sharing and cross-basis independence of priors.
- **Identifier Stability:** Hardcoded regression anchor for the full model.
- **Serialization Round-Trip:** dict/from_dict preserves prior count and path structure.
- **Model Info:** Human-readable info string contains expected component names.
"""

import autofit as af
import autogalaxy as ag

"""
__Galaxy 0 Composition__

Galaxy 0 uses an MGE bulge with ``gaussian_per_basis=2``, which gives each
basis independent ellipticity components while sharing a common centre.

Expected free parameters:
- bulge: 2 (shared centre) + 2*2 (ell_comps per basis) = 6
- total galaxy_0: 6
"""

mask_radius = 3.0

bulge_0 = ag.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
)

assert (
    bulge_0.prior_count == 6
), f"Galaxy 0 bulge prior_count: expected 6, got {bulge_0.prior_count}"

galaxy_0 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_0)
assert (
    galaxy_0.prior_count == 6
), f"Galaxy 0 prior_count: expected 6, got {galaxy_0.prior_count}"

print("Galaxy 0 composition: PASSED")

"""
__Galaxy 1 Composition__

Galaxy 1 uses a single-basis MGE (``gaussian_per_basis=1``).

Expected free parameters:
- bulge: 2 (centre) + 2 (ell_comps) = 4
"""

bulge_1 = ag.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    centre_prior_is_uniform=False,
)

assert (
    bulge_1.prior_count == 4
), f"Galaxy 1 bulge prior_count: expected 4, got {bulge_1.prior_count}"

galaxy_1 = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge_1)
assert (
    galaxy_1.prior_count == 4
), f"Galaxy 1 prior_count: expected 4, got {galaxy_1.prior_count}"

print("Galaxy 1 composition: PASSED")

"""
__Full Model__

The full model is an ``af.Collection`` wrapping both galaxies (both at the same
redshift — autogalaxy has no ray-tracing). The total free parameter count
should be the sum of the per-galaxy counts.
"""

model = af.Collection(galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1))

assert (
    model.prior_count == 10
), f"Full model prior_count: expected 10, got {model.prior_count}"

paths = model.unique_prior_paths

assert len(paths) == 10, f"Expected 10 unique prior paths, got {len(paths)}"

for path in paths:
    assert path[0] == "galaxies", f"Top-level path should be 'galaxies': {path}"
    assert path[1] in ("galaxy_0", "galaxy_1"), f"Galaxy name unexpected: {path}"

galaxy_0_paths = [p for p in paths if p[1] == "galaxy_0"]
galaxy_1_paths = [p for p in paths if p[1] == "galaxy_1"]
assert (
    len(galaxy_0_paths) == 6
), f"Expected 6 galaxy_0 paths, got {len(galaxy_0_paths)}"
assert (
    len(galaxy_1_paths) == 4
), f"Expected 4 galaxy_1 paths, got {len(galaxy_1_paths)}"

print("Full model composition: PASSED")

"""
__MGE Prior Identity__

This is the critical structural test. Within a basis, all Gaussians must share
the same centre and ell_comps prior objects (by Python identity). Across bases,
centres are shared (default ``centre_per_basis=False``) but ell_comps are
independent.

This catches the exact class of bug where a refactor moves prior construction
across a loop boundary, silently collapsing ``gaussian_per_basis > 1`` into a
degenerate model.
"""

gaussians = list(bulge_0.profile_list)
assert len(gaussians) == 40, f"Expected 40 Gaussians, got {len(gaussians)}"

basis_0 = gaussians[:20]
basis_1 = gaussians[20:]

for g in basis_0:
    assert (
        g.centre.centre_0 is basis_0[0].centre.centre_0
    ), "Centres within basis 0 must be shared by identity"
    assert g.centre.centre_1 is basis_0[0].centre.centre_1

for g in basis_1:
    assert (
        g.centre.centre_0 is basis_1[0].centre.centre_0
    ), "Centres within basis 1 must be shared by identity"

assert (
    basis_0[0].centre.centre_0 is basis_1[0].centre.centre_0
), "Centres across bases must be shared (centre_per_basis=False)"

for g in basis_0:
    assert (
        g.ell_comps.ell_comps_0 is basis_0[0].ell_comps.ell_comps_0
    ), "ell_comps within basis 0 must be shared by identity"

for g in basis_1:
    assert (
        g.ell_comps.ell_comps_0 is basis_1[0].ell_comps.ell_comps_0
    ), "ell_comps within basis 1 must be shared by identity"

assert basis_0[0].ell_comps.ell_comps_0 is not basis_1[0].ell_comps.ell_comps_0, (
    "ell_comps across bases must be INDEPENDENT (this is the whole point of "
    "gaussian_per_basis > 1)"
)
assert basis_0[0].ell_comps.ell_comps_1 is not basis_1[0].ell_comps.ell_comps_1

print("MGE prior identity: PASSED")

"""
__Identifier Stability__

The identifier is an md5 hash that determines the output folder. If it changes
after a PyAutoFit refactor, existing users' results folders will no longer match.

This hardcoded value is the regression anchor. Update it only if the identifier
change is intentional (e.g. a deliberate algorithm change). An accidental change
means a refactor has silently altered model composition.
"""

from autofit.non_linear.paths.directory import DirectoryPaths

paths_obj = DirectoryPaths()
paths_obj.model = model

identifier = paths_obj.identifier

assert len(identifier) == 32
assert identifier.isalnum()

assert identifier == "a6eb928ed9a1fb92d0c18cf5443af4a6", (
    f"REGRESSION: multi-galaxy MGE model identifier changed from expected value. "
    f"Got '{identifier}'. If this is intentional, update the expected value. "
    f"If not, a PyAutoFit or PyAutoGalaxy refactor has silently altered how "
    f"this model is composed — this would break backwards compatibility for "
    f"every autogalaxy user running MGE models."
)

print("Identifier stability: PASSED")

"""
__Serialization Round-Trip__

``model.dict()`` → ``from_dict()`` must preserve prior count and path structure.
"""

d = model.dict()
restored = af.Collection.from_dict(d)

assert (
    restored.prior_count == model.prior_count
), f"prior_count changed after round-trip: {restored.prior_count} vs {model.prior_count}"

assert (
    restored.unique_prior_paths == model.unique_prior_paths
), "Path structure changed after serialization round-trip"

print("Serialization round-trip: PASSED")

"""
__Model Info__

The human-readable ``model.info`` string should contain the expected component
class names.
"""

info = model.info

for keyword in ["Galaxy", "Basis"]:
    assert keyword in info, f"model.info missing expected keyword: {keyword}"

print("Model info: PASSED")

"""
__Gaussian Per Basis Variants__

Verify that different ``gaussian_per_basis`` values produce the correct prior
counts and that the structural guarantees hold for each.
"""

for gpb in [1, 2, 3]:
    m = ag.model_util.mge_model_from(
        mask_radius=3.0,
        total_gaussians=10,
        gaussian_per_basis=gpb,
    )

    if gpb == 1:
        expected = 4
    else:
        expected = 2 + 2 * gpb

    assert (
        m.prior_count == expected
    ), f"gaussian_per_basis={gpb}: expected {expected}, got {m.prior_count}"

    gs = list(m.profile_list)
    assert len(gs) == 10 * gpb

    for basis_idx in range(gpb):
        basis_slice = gs[basis_idx * 10 : (basis_idx + 1) * 10]
        for g in basis_slice:
            assert g.centre.centre_0 is basis_slice[0].centre.centre_0

    if gpb > 1:
        assert (
            gs[0].ell_comps.ell_comps_0 is not gs[10].ell_comps.ell_comps_0
        ), f"gaussian_per_basis={gpb}: ell_comps should be independent across bases"

print("Gaussian per basis variants: PASSED")

"""
__Spherical MGE__

``use_spherical=True`` removes ell_comps entirely. Only centre priors are free.
"""

m_sph = ag.model_util.mge_model_from(
    mask_radius=3.0,
    total_gaussians=10,
    gaussian_per_basis=2,
    use_spherical=True,
)

assert (
    m_sph.prior_count == 2
), f"Spherical MGE (shared centre): expected 2, got {m_sph.prior_count}"

m_sph_cpb = ag.model_util.mge_model_from(
    mask_radius=3.0,
    total_gaussians=10,
    gaussian_per_basis=2,
    use_spherical=True,
    centre_per_basis=True,
)

assert (
    m_sph_cpb.prior_count == 4
), f"Spherical MGE (per-basis centre): expected 4, got {m_sph_cpb.prior_count}"

print("Spherical MGE: PASSED")

"""
__Centre Per Basis__

``centre_per_basis=True`` gives each basis independent centre priors.
"""

m_cpb = ag.model_util.mge_model_from(
    mask_radius=3.0,
    total_gaussians=10,
    gaussian_per_basis=2,
    centre_per_basis=True,
)

assert (
    m_cpb.prior_count == 8
), f"centre_per_basis=True: expected 8, got {m_cpb.prior_count}"

gs = list(m_cpb.profile_list)
assert (
    gs[0].centre.centre_0 is not gs[10].centre.centre_0
), "centre_per_basis=True: centres should be independent across bases"

print("Centre per basis: PASSED")

"""
__Centre Fixed__

``centre_fixed=(0.0, 0.0)`` fixes all centres, leaving only ell_comps as free.
"""

m_cf = ag.model_util.mge_model_from(
    mask_radius=3.0,
    total_gaussians=10,
    gaussian_per_basis=2,
    centre_fixed=(0.0, 0.0),
)

assert m_cf.prior_count == 4, f"centre_fixed: expected 4, got {m_cf.prior_count}"

print("Centre fixed: PASSED")

"""
__Summary__
"""

print()
print("All multi-galaxy MGE integration tests: PASSED")
