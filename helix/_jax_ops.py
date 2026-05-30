"""Back-compat shim — TPC group ops moved to helix.tpc.coherent_ops_jax,
wavelet matmul ops to helix.core.wavelet_ops_jax / helix.core.dwt_matrix.
"""
from helix.tpc.coherent_ops_jax import (
    group_median, broadcast_groups, signal_mask, temporal_dilate,
    masked_group_mean, xblock_kernel, pad_to_groups, pad_mask_to_groups,
)

__all__ = [
    "group_median", "broadcast_groups", "signal_mask", "temporal_dilate",
    "masked_group_mean", "xblock_kernel", "pad_to_groups", "pad_mask_to_groups",
]
