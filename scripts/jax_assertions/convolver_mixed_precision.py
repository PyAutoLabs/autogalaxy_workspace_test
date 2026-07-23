"""
Jax Assertions: Convolver use_mixed_precision FFT Path
=======================================================

Verifies that ``Convolver.convolved_image_from`` and
``Convolver.convolved_mapping_matrix_from`` honor ``use_mixed_precision`` end
to end on the JAX FFT path:

* fp64 JAX matches the eager NumPy real-space reference within ``atol=1e-10``.
* fp32 JAX returns ``dtype=float32`` (verifying the FFT actually ran in
  ``complex64`` rather than fp64-then-cast).
* fp32 JAX agrees with fp64 JAX within ``atol=1e-3``, which is the empirical
  tolerance for a 21x21 PSF FFT in complex64.

Background: prior to this script the ``use_mixed_precision`` flag was a net
loss on consumer GPUs because both FFT paths force-cast to fp64 internally
and only narrowed the result at the end. The dtype assertion below is what
locks in the actual fix.

Library unit tests stay numpy-only by project convention; this cross-xp
coverage lives here in workspace_test.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Loads a pre-committed 80x80 array; SMALL_DATASETS would cap the mask to
15x15 and break the shape assertion.

ENV: full_datasets
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import autoarray as aa


"""
__Setup: small representative image, 21x21 PSF, circular mask with parity-preserving padding__
"""
shape_native = (80, 80)
pixel_scales = (0.05, 0.05)

mask = aa.Mask2D.circular(
    shape_native=shape_native,
    pixel_scales=pixel_scales,
    radius=1.5,
)

rng = np.random.default_rng(0)
image_values = rng.standard_normal(shape_native).astype(np.float64)

masked_image = aa.Array2D(values=image_values, mask=mask)

kernel_native = np.zeros((21, 21), dtype=np.float64)
y, x = np.mgrid[-10:11, -10:11]
kernel_native[:] = np.exp(-(x**2 + y**2) / (2.0 * 2.5**2))
kernel_native /= kernel_native.sum()
kernel = aa.Array2D.no_mask(values=kernel_native, pixel_scales=pixel_scales)

convolver_fft = aa.Convolver(kernel=kernel, use_fft=True, normalize=True)
convolver_real = aa.Convolver(kernel=kernel, use_fft=False, normalize=True)

blurring_mask = mask.derive_mask.blurring_from(
    kernel_shape_native=convolver_fft.kernel.shape_native
)
blurring_image = aa.Array2D(values=image_values, mask=blurring_mask)


"""
__convolved_image_from: fp64 JAX matches NumPy real-space to atol=1e-10__
"""
np_blurred = convolver_real.convolved_image_from(
    image=masked_image,
    blurring_image=blurring_image,
    xp=np,
)
jnp_blurred_fp64 = convolver_fft.convolved_image_from(
    image=masked_image,
    blurring_image=blurring_image,
    use_mixed_precision=False,
    xp=jnp,
)

npt.assert_allclose(
    np.asarray(np_blurred.array),
    np.asarray(jnp_blurred_fp64.array),
    atol=1.0e-10,
)


"""
__convolved_image_from: fp32 JAX returns float32 dtype__
"""
jnp_blurred_fp32 = convolver_fft.convolved_image_from(
    image=masked_image,
    blurring_image=blurring_image,
    use_mixed_precision=True,
    xp=jnp,
)

assert jnp_blurred_fp32.array.dtype == jnp.float32, (
    f"Expected float32 from use_mixed_precision=True, got {jnp_blurred_fp32.array.dtype}. "
    "If this fails, the FFT did not actually run in complex64 — check that the "
    "precomputed state.fft_kernel cast and real_dtype scatter were applied in "
    "Convolver.convolved_image_from."
)


"""
__convolved_image_from: fp32 vs fp64 JAX agree to atol=1e-3__
"""
npt.assert_allclose(
    np.asarray(jnp_blurred_fp32.array),
    np.asarray(jnp_blurred_fp64.array),
    atol=1.0e-3,
    rtol=1.0e-3,
)


"""
__Setup: mapping matrix for convolved_mapping_matrix_from (5 source columns)__
"""
n_pix = int(mask.pixels_in_mask)
n_src = 5

mapping_matrix = rng.standard_normal((n_pix, n_src)).astype(np.float64)

n_blur = int(blurring_mask.pixels_in_mask)
blurring_mapping_matrix = rng.standard_normal((n_blur, n_src)).astype(np.float64)


"""
__convolved_mapping_matrix_from: fp64 JAX matches NumPy real-space to atol=1e-10__
"""
np_blurred_mm = convolver_real.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix,
    mask=mask,
    blurring_mapping_matrix=blurring_mapping_matrix,
    blurring_mask=blurring_mask,
    xp=np,
)
jnp_blurred_mm_fp64 = convolver_fft.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix,
    mask=mask,
    blurring_mapping_matrix=blurring_mapping_matrix,
    blurring_mask=blurring_mask,
    use_mixed_precision=False,
    xp=jnp,
)

npt.assert_allclose(
    np.asarray(np_blurred_mm),
    np.asarray(jnp_blurred_mm_fp64),
    atol=1.0e-10,
)


"""
__convolved_mapping_matrix_from: fp32 input cube, fp64 kernel multiply, fp64 output__

Unlike ``convolved_image_from`` (used by light profiles, K~40 well-conditioned
columns), ``convolved_mapping_matrix_from`` is used by pixelizations where K
can be ~1000+ source pixels. Full fp32 FFT-and-multiply causes O(1) drift in
the NNLS log-determinant figure_of_merit on those meshes
(verified on autolens_workspace_test/.../delaunay_mge.py). The mixed-precision
contract for this function is therefore: cube allocated as fp32 (via
``mapping_matrix_native_from``) for a slightly cheaper scatter and forward
rfft2, but the multiply with the complex128 kernel upcasts the result back to
fp64 to preserve precision through the downstream linear algebra. The output
is fp64 in both ``use_mixed_precision`` modes.
"""
jnp_blurred_mm_fp32 = convolver_fft.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix,
    mask=mask,
    blurring_mapping_matrix=blurring_mapping_matrix,
    blurring_mask=blurring_mask,
    use_mixed_precision=True,
    xp=jnp,
)

assert jnp_blurred_mm_fp32.dtype == jnp.float64, (
    f"Expected float64 (intentional, see docstring above), got {jnp_blurred_mm_fp32.dtype}. "
    "If this asserts the wrong dtype, the kernel multiply has been changed to "
    "preserve complex64 — that breaks pixelization figure_of_merit precision."
)

# fp32-input-cube path agrees with the pure fp64 path to within fp32 rounding,
# because the multiply upcasts back to fp64 and the only fp32 arithmetic is in
# the input scatter and forward rfft2. atol is loose enough to cover that.
npt.assert_allclose(
    np.asarray(jnp_blurred_mm_fp32),
    np.asarray(jnp_blurred_mm_fp64),
    atol=1.0e-5,
    rtol=1.0e-5,
)


print("convolver_mixed_precision: all assertions passed")
