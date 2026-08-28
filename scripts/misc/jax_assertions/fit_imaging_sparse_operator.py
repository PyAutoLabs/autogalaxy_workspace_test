"""
Jax Assertions: FitImaging Sparse Operator
==========================================

Cross-implementation parity check for the ``ImagingSparseOperator`` path inside
an autogalaxy ``FitImaging`` fit — the autogalaxy counterpart of
``autolens_workspace_test/scripts/misc/jax_assertions/fit_imaging_sparse_operator.py``,
which covers the same operator through a multi-plane ``Tracer``. This script
covers the plain multi-galaxy route, where the linear objects come straight
from a galaxy list with no ray tracing in between.

A galaxy list whose linear objects are a MIXED list — a linear light profile
func list (``lp_linear.Sersic``) plus a ``Mapper`` (``RectangularUniform`` +
``Constant`` regularization) — is fitted twice:

- via the standard mapping-matrix path (``InversionImagingMapping``), and
- via ``masked_dataset.apply_sparse_operator()`` (``InversionImagingSparse``).

The two must agree on ``curvature_matrix``, ``data_vector``,
``regularization_matrix``, ``reconstruction`` and ``log_evidence``.

Sibling of ``fit_interferometer_sparse_operator.py`` in this directory, which
runs the equivalent check through ``InversionInterferometerSparse`` (the class
PyAutoArray #499 / #500 taught to handle func lists and multiple mappers).

The noise-map is deliberately NON-uniform: a unit noise-map makes the
inverse-variance weighting the identity and would hide a weighting bug in the
sparse operator.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
The sparse operator is the JAX accelerator path this script exists to gate, so
JAX must stay enabled. The dataset is built in-memory at 15 x 15, so the
SMALL_DATASETS cap is irrelevant either way.

ENV: jax
"""

import numpy as np

import autoarray as aa
import autogalaxy as ag


"""
__Dataset__

A small in-memory `Imaging` dataset simulated from two Sersic profiles, then
given a per-pixel noise-map and masked down to 45 image pixels.
"""
rng = np.random.default_rng(1234)

grid = ag.Grid2D.uniform(shape_native=(15, 15), pixel_scales=0.2)

psf = ag.Convolver.from_gaussian(
    shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
)

galaxy_simulate = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(centre=(0.0, 0.0), effective_radius=0.3),
    disk=ag.lp.Sersic(centre=(0.1, 0.1), effective_radius=0.6),
)

simulator = ag.SimulatorImaging(
    exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
)

dataset = simulator.via_galaxies_from(galaxies=[galaxy_simulate], grid=grid)
dataset.sub_size = 2

dataset.noise_map = ag.Array2D(
    values=rng.uniform(0.5, 2.0, size=dataset.data.shape_native),
    mask=dataset.data.mask,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
)

masked_dataset = dataset.apply_mask(mask=mask)
masked_dataset_sparse_operator = masked_dataset.apply_sparse_operator()

"""
__Galaxies__

One galaxy contributes the linear light profile func list, the other the
mapper, so the inversion's linear object list is mixed — the func-list diagonal
block and the func-list/mapper off-diagonals are both exercised.
"""
galaxies = [
    ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.Sersic(centre=(0.0, 0.0), effective_radius=0.3),
    ),
    ag.Galaxy(
        redshift=0.5,
        pixelization=ag.Pixelization(
            mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
            regularization=ag.reg.Constant(coefficient=1.0),
        ),
    ),
]

"""
The positive-only (NNLS) solver is the production default, but on a fit this
small its active set clamps most reconstructed values to exactly `0.0`, which
would make the `reconstruction` assertion vacuous. The direct solve keeps every
reconstructed value non-trivial; the curvature matrix and data vector that feed
it are solver-independent.
"""
settings = ag.Settings(use_positive_only_solver=False)

fit_mapping = ag.FitImaging(
    dataset=masked_dataset,
    galaxies=galaxies,
    settings=settings,
)

fit_sparse_operator = ag.FitImaging(
    dataset=masked_dataset_sparse_operator,
    galaxies=galaxies,
    settings=settings,
)

inversion_mapping = fit_mapping.inversion
inversion_sparse_operator = fit_sparse_operator.inversion

assert isinstance(inversion_mapping, aa.InversionImagingMapping), (
    f"expected the un-operated dataset to route to InversionImagingMapping, got "
    f"{type(inversion_mapping).__name__}"
)
assert isinstance(inversion_sparse_operator, aa.InversionImagingSparse), (
    f"expected the sparse-operator dataset to route to InversionImagingSparse, "
    f"got {type(inversion_sparse_operator).__name__}"
)

linear_obj_types = [type(obj).__name__ for obj in inversion_mapping.linear_obj_list]
assert any(
    "FuncList" in linear_obj_type for linear_obj_type in linear_obj_types
), f"no linear func list in {linear_obj_types}"
assert any(
    "Mapper" in linear_obj_type for linear_obj_type in linear_obj_types
), f"no mapper in {linear_obj_types}"

"""
__Shape__

Before PyAutoArray #500 the sparse interferometer `curvature_matrix` was the
single-mapper diagonal block ALONE; the imaging sparse path never had that bug,
but the same guard is cheap here and keeps a shape regression from surfacing as
an unreadable broadcast error.
"""
for quantity in ("curvature_matrix", "data_vector", "reconstruction"):
    shape_sparse_operator = np.asarray(
        getattr(inversion_sparse_operator, quantity)
    ).shape
    shape_mapping = np.asarray(getattr(inversion_mapping, quantity)).shape
    assert shape_sparse_operator == shape_mapping, (
        f"sparse-operator {quantity} has shape {shape_sparse_operator}, mapping "
        f"path has {shape_mapping} — the sparse path is not forming every "
        f"linear-object block"
    )

"""
__Assertions__

Every quantity is compared as a maximum ABSOLUTE difference normalised by the
largest entry of the mapping-path array — a plain elementwise relative error is
meaningless on a curvature matrix whose entries span many orders of magnitude.
The measured values are printed so drift is visible in the smoke log rather
than only on failure.

`cond(F + H) ~ 5e1` here, so nothing is amplified by the solve and every
quantity agrees at float64 round-off. Measured maxima:

    curvature_matrix       1.6e-16
    data_vector            1.1e-16
    regularization_matrix  0.0
    reconstruction         1.3e-15
    log_evidence           0.0

`rtol=1e-11` therefore leaves >1e4 margin on the worst of them while still
failing hard on a dropped block, which is an O(1) relative error.
"""


def max_relative_error(sparse_operator, mapping):
    sparse_operator = np.asarray(sparse_operator)
    mapping = np.asarray(mapping)
    scale = float(np.abs(mapping).max())
    scale = scale if scale > 0.0 else 1.0
    return float(np.abs(sparse_operator - mapping).max()) / scale


error_dict = {
    "curvature_matrix": max_relative_error(
        inversion_sparse_operator.curvature_matrix,
        inversion_mapping.curvature_matrix,
    ),
    "data_vector": max_relative_error(
        inversion_sparse_operator.data_vector,
        inversion_mapping.data_vector,
    ),
    "regularization_matrix": max_relative_error(
        inversion_sparse_operator.regularization_matrix,
        inversion_mapping.regularization_matrix,
    ),
    "reconstruction": max_relative_error(
        inversion_sparse_operator.reconstruction,
        inversion_mapping.reconstruction,
    ),
    "log_evidence": abs(
        float(fit_sparse_operator.log_evidence) - float(fit_mapping.log_evidence)
    )
    / abs(float(fit_mapping.log_evidence)),
}

for quantity, error in error_dict.items():
    print(f"  {quantity}: max relative error {error:.2e}")

for quantity, error in error_dict.items():
    assert error < 1.0e-11, f"{quantity} max relative error {error:.3e} exceeds 1e-11"

print("fit_imaging_sparse_operator: all assertions passed")
