import argparse
from pathlib import Path
import torch
from transformers.cache_utils import DynamicCache
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


class KVCloak:
    def __init__(
        self,
        kvcloak_config: List[List[List[Dict[str, torch.Tensor]]]],
        dtype: torch.dtype,
        fused: bool,
        need_ratio: bool,
        add_a: bool,
    ):
        """
        Initialize KVCloak instance.

        Args:
            kvcloak_config (List): Nested list containing keys for all layers and heads.
            dtype (torch.dtype): Data type used for computation.
            fused (bool): Flag indicating whether to use fused operations.
                          When True, M matrix operations are skipped (applied via fusion.py).
            need_ratio (bool): Flag indicating whether to apply scaling ratios.
            add_a (bool): Flag indicating whether to add staggered markers:
                          every row of A carries a marker a_i ~ U(3i*t, (3i+0.5)*t),
                          stripped at deobfuscation by value-range classification
                          (j = floor((row_max + t) / (3t))).
        """
        if not kvcloak_config or not kvcloak_config[0] or not kvcloak_config[0][0]:
            raise ValueError("kvcloak_config cannot be empty or have invalid structure.")

        self.kvcloak_config = kvcloak_config
        self.dtype = dtype
        self.fused = fused
        self.need_ratio = need_ratio
        self.add_a = add_a
        self.b = self.kvcloak_config[0][0][0]["S"].shape[0]
        self.num_layers = len(self.kvcloak_config)
        self.num_heads = len(self.kvcloak_config[0]) if self.num_layers > 0 else 0

        # Cache precomputed tensors for different devices
        self._device_cache: Dict[torch.device, Dict] = {}


    def _prepare_device_tensors(self, device: torch.device):
        """
        Prepare and cache all reusable tensors for the specified device.
        This method is called only once when encountering a new device for the first time.
        """
        if device in self._device_cache:
            return

        # Used to store precomputed tensors for each layer
        obf_tensors_by_layer = []
        inv_tensors_by_layer = []

        for layer_idx in range(self.num_layers):
            # Prepare lists for keys (K) and values (V)
            S_list_k, M_list_k, A_list_k = [], [], []
            S_list_v, M_list_v, A_list_v = [], [], []
            threshold_list_k, threshold_list_v = [], []

            inv_S_list_k, inv_M_list_k, a_list_k = [], [], []
            inv_S_list_v, inv_M_list_v, a_list_v = [], [], []

            for head_idx in range(self.num_heads):
                config_k = self.kvcloak_config[layer_idx][head_idx][0]
                config_v = self.kvcloak_config[layer_idx][head_idx][1]

                # --- Prepare obfuscation matrices ---
                S_k, M_k, A_k = self._get_SMA_obf(config_k, device)
                S_list_k.append(S_k)
                M_list_k.append(M_k)
                A_list_k.append(A_k)
                threshold_list_k.append(config_k["theta"] * config_k["M_ratio"] * 1.42)

                S_v, M_v, A_v = self._get_SMA_obf(config_v, device)
                S_list_v.append(S_v)
                M_list_v.append(M_v)
                A_list_v.append(A_v)
                threshold_list_v.append(config_v["theta"] * config_v["M_ratio"] * 1.42)

                # --- Prepare deobfuscation matrices ---
                inv_S_k, inv_M_k, a_k = self._get_SMA_inv(config_k, device)
                inv_S_list_k.append(inv_S_k)
                inv_M_list_k.append(inv_M_k)
                a_list_k.append(a_k)

                inv_S_v, inv_M_v, a_v = self._get_SMA_inv(config_v, device)
                inv_S_list_v.append(inv_S_v)
                inv_M_list_v.append(inv_M_v)
                a_list_v.append(a_v)

            d = M_k.shape[-1]

            threshold_k_tensor = torch.tensor(
                threshold_list_k, device=device, dtype=self.dtype
            ).view(1, self.num_heads, 1, 1)
            threshold_v_tensor = torch.tensor(
                threshold_list_v, device=device, dtype=self.dtype
            ).view(1, self.num_heads, 1, 1)

            obf_tensors_by_layer.append(
                {
                    "S_k_batch": torch.stack(S_list_k).view(
                        1, self.num_heads, 1, self.b, self.b
                    ),
                    "M_k_batch": torch.stack(M_list_k).view(1, self.num_heads, 1, d, d),
                    "A_k_batch": torch.stack(A_list_k).view(
                        1, self.num_heads, 1, self.b, d
                    ),
                    "S_v_batch": torch.stack(S_list_v).view(
                        1, self.num_heads, 1, self.b, self.b
                    ),
                    "M_v_batch": torch.stack(M_list_v).view(1, self.num_heads, 1, d, d),
                    "A_v_batch": torch.stack(A_list_v).view(
                        1, self.num_heads, 1, self.b, d
                    ),
                    "thresholds_k": threshold_k_tensor,
                    "thresholds_v": threshold_v_tensor,
                }
            )

            inv_tensors_by_layer.append(
                {
                    "inverse_S_k_batch": torch.stack(inv_S_list_k).view(
                        1, self.num_heads, 1, self.b, self.b
                    ),
                    "inverse_M_k_batch": torch.stack(inv_M_list_k).view(
                        1, self.num_heads, 1, d, d
                    ),
                    "a_k_list": torch.stack(a_list_k),
                    "inverse_S_v_batch": torch.stack(inv_S_list_v).view(
                        1, self.num_heads, 1, self.b, self.b
                    ),
                    "inverse_M_v_batch": torch.stack(inv_M_list_v).view(
                        1, self.num_heads, 1, d, d
                    ),
                    "a_v_list": torch.stack(a_list_v),
                    "thresholds_k": threshold_k_tensor,
                    "thresholds_v": threshold_v_tensor,
                }
            )

        self._device_cache[device] = {
            "obf": obf_tensors_by_layer,
            "inv": inv_tensors_by_layer,
        }

    @staticmethod
    def _get_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
        """Static method: Create rotation matrix from angle vector."""
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)

        batch_size, d_half = angles.shape
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        M = torch.zeros(
            (batch_size, 2 * d_half, 2 * d_half),
            device=angles.device,
            dtype=angles.dtype,
        )

        diag_cos = cos_a.repeat(1, 2)
        M = M + torch.diag_embed(diag_cos, offset=0)
        M = M + torch.diag_embed(sin_a, offset=d_half)
        M = M + torch.diag_embed(-sin_a, offset=-d_half)

        return M.squeeze(0)

    def _marker_table(
        self, config: Dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Get the staggered marker lookup table from config["A"].

        Returns a (b+1, d) tensor whose row 0 is zeros and row i holds marker
        a_i with elements uniform in (3i*t, (3i+0.5)*t), t = head threshold.
        The table is generated by get_kvcloak_config and saved with the
        config, so obfuscation and deobfuscation (even in different
        processes) share the identical table.
        """
        return config["A"].to(device=device, dtype=self.dtype)

    def _get_SMA_obf(
        self, config: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get S, M, A matrices for obfuscation."""
        S_ratios = config["S_ratios"].to(device, self.dtype)
        S = config["S"].to(device, self.dtype)
        m_ratios = config["M_ratios"].to(device, self.dtype)
        m_angles = config["M_angles"].to(device, self.dtype)

        M = self._get_rotation_matrix(m_angles).to(device, self.dtype)
        if self.need_ratio:
            S = S * S_ratios
            doubled_m_ratios = torch.cat((m_ratios, m_ratios))
            M = M * doubled_m_ratios

        # every row of A carries its own staggered marker a_i
        A = self._marker_table(config, device)[1:]
        return S, M, A

    def _get_SMA_inv(
        self, config: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get inverse S, M and the (b+1, d) staggered marker lookup table."""
        S_ratios = config["S_ratios"].to(device, self.dtype)
        S = config["S"].to(device, self.dtype)
        m_ratios = config["M_ratios"].to(device, self.dtype)
        m_angles = config["M_angles"].to(device, self.dtype)

        M = self._get_rotation_matrix(m_angles).to(device, self.dtype)

        if self.need_ratio:
            inverse_S = (S / S_ratios).T
            doubled_m_ratios = torch.cat((m_ratios, m_ratios))
            inverse_M = M.T / doubled_m_ratios
        else:
            inverse_S = S.T
            inverse_M = M.T

        return inverse_S, inverse_M, self._marker_table(config, device)

    def _strip_staggered_markers_(
        self,
        ws: torch.Tensor,
        a_table: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor:
        """Strip staggered per-row markers from ws (B, H, num_blocks, b, d), in place.

        After inverse_S, each row holds data (or padding) plus the marker a_j
        of its original in-block position j (the row permutation P is unknown).
        Data lies in (-t, t) and padding in (1.1t, 1.5t), so a row carrying
        marker j has row_max in ((3j-1)t, (3j+2)t) and
        j = floor((row_max + t) / (3t)) classifies the marker exactly; the
        marker is then subtracted via a one-hot contraction with the lookup
        table (vectorized "gather", measured faster than a boolean-mask
        subtract).
        """
        thr = thresholds.view(1, self.num_heads, 1, 1)
        row_max = ws.amax(dim=-1)  # (B, H, num_blocks, b)
        j = torch.floor((row_max + thr) / (3.0 * thr)).clamp_(0, self.b).long()
        onehot = torch.nn.functional.one_hot(j, self.b + 1).to(ws.dtype)
        a_sel = torch.einsum("hkd,bhnqk->bhnqd", a_table, onehot)
        ws.sub_(a_sel)
        return ws

    @staticmethod
    def _remove_padding_rows(
        padded_tensor: torch.Tensor, threshold: torch.Tensor, b: int
    ) -> torch.Tensor:
        """Remove padding rows from the last block via per-row classification.

        After marker removal, a padding row has ALL elements above the head
        threshold (padding is filled in (1.1*t, 1.5*t)), while data rows stay
        below it (|data| <= theta < t thanks to the theta_ratio margin).
        Rows are classified per batch/head with per-head thresholds, and a
        majority vote across batch/heads keeps the removal mask uniform.
        """
        batch_size, num_heads, seq_len, d = padded_tensor.shape
        thr = threshold.view(1, num_heads, 1, 1)
        last_block = padded_tensor[:, :, -b:, :]
        is_pad = (last_block > thr).all(dim=-1)  # (B, H, b)
        is_pad_pos = is_pad.float().mean(dim=(0, 1)) > 0.5  # (b,)
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=padded_tensor.device)
        keep_mask[seq_len - b:] = ~is_pad_pos
        return padded_tensor[:, :, keep_mask, :]

    def obfuscate(self, orig_kvcache: DynamicCache) -> DynamicCache:
        device = orig_kvcache[0][0].device
        self._prepare_device_tensors(device)
        cached_obf_tensors = self._device_cache[device]["obf"]
        kvcloaked_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, (K_orig, V_orig) in enumerate(orig_kvcache):
            batch_size, _, seq_len, d = K_orig.shape
            K_dtype = K_orig.dtype

            layer_tensors = cached_obf_tensors[layer_idx]
            S_k_batch = layer_tensors["S_k_batch"]
            M_k_batch = layer_tensors["M_k_batch"]
            A_k_batch = layer_tensors["A_k_batch"]
            S_v_batch = layer_tensors["S_v_batch"]
            M_v_batch = layer_tensors["M_v_batch"]
            A_v_batch = layer_tensors["A_v_batch"]
            thresholds_k = layer_tensors["thresholds_k"]
            thresholds_v = layer_tensors["thresholds_v"]

            rem = seq_len % self.b
            if rem > 0:
                padding_len = self.b - rem
                padded_seq_len = seq_len + padding_len

                # OPTIMIZATION: Create tensor with final size directly, avoiding torch.cat and intermediate padding tensors
                K_padded = torch.empty(
                    (batch_size, self.num_heads, padded_seq_len, d),
                    device=device,
                    dtype=K_dtype,
                )
                K_padded[:, :, :seq_len, :] = K_orig
                # Random padding in (1.1*t, 1.5*t): a constant fill of 1.5*t
                # would embed an exactly-known component (1.5*t * 1_d) that an
                # attacker could reproduce element-wise. Deobfuscation detects
                # padding rows by range (all elements > t), so the random
                # values never need to be reproduced.
                # For the non-fused path this pre-fill is overwritten after the
                # M rotation (see below): markers and padding are classified
                # by
                # value ranges, which requires the padding to live in the
                # post-rotation (kM) domain. There the marker (if it lands on
                # a padding row) satisfies pad + a > 2t exactly, so marker
                # stripping is never ambiguous.
                if self.fused:
                    pad_k = K_padded[:, :, seq_len:, :]
                    pad_k.uniform_(1.1, 1.5)
                    pad_k.mul_(thresholds_k.to(K_dtype))

                V_padded = torch.empty(
                    (batch_size, self.num_heads, padded_seq_len, d),
                    device=device,
                    dtype=K_dtype,
                )
                V_padded[:, :, :seq_len, :] = V_orig
                if self.fused:
                    pad_v = V_padded[:, :, seq_len:, :]
                    pad_v.uniform_(1.1, 1.5)
                    pad_v.mul_(thresholds_v.to(K_dtype))
            else:
                padded_seq_len = seq_len
                # The fused path writes results into a separate workspace and only
                # reads K/V, so the caller's cache can be used directly. The
                # non-fused path writes out=K_blocks, which must not alias the
                # caller's tensors.
                if self.fused:
                    K_padded, V_padded = K_orig, V_orig
                else:
                    K_padded, V_padded = K_orig.clone(), V_orig.clone()

            num_blocks = padded_seq_len // self.b
            K_blocks = K_padded.view(
                batch_size, self.num_heads, num_blocks, self.b, d
            ).to(self.dtype)
            V_blocks = V_padded.view(
                batch_size, self.num_heads, num_blocks, self.b, d
            ).to(self.dtype)

            # OPTIMIZATION: Preallocate a workspace to store intermediate matmul results
            K_workspace = torch.empty_like(K_blocks)
            V_workspace = torch.empty_like(V_blocks)

            perm_indices = torch.argsort(
                torch.rand(num_blocks, self.b, device=device), dim=1
            )
            perm_matrices = torch.eye(self.b, device=device, dtype=self.dtype)[
                perm_indices
            ]

            # --- Key Obfuscation (fused: out = Sp@X + Sp@A) ---
            if not self.fused:
                torch.matmul(K_blocks, M_k_batch, out=K_workspace)
                if rem > 0:
                    # fill padding in the kM domain so that value-range based
                    # marker/padding classification at deobfuscation is exact
                    pad_ws = K_workspace.view(
                        batch_size, self.num_heads, num_blocks, self.b, d
                    )[:, :, -1, rem:, :]
                    pad_ws.uniform_(1.1, 1.5)
                    pad_ws.mul_(thresholds_k.to(K_workspace.dtype))
                if self.add_a:
                    K_workspace.add_(A_k_batch)
                Sp_k_batch = torch.matmul(S_k_batch, perm_matrices)
                torch.matmul(Sp_k_batch, K_workspace, out=K_blocks)
            else:
                Sp_k_batch = torch.matmul(S_k_batch, perm_matrices)
                num_blocks_ = K_blocks.shape[2]
                flat = batch_size * self.num_heads * num_blocks_
                Sp_exp = Sp_k_batch.expand(
                    batch_size, self.num_heads, num_blocks_, self.b, self.b
                ).reshape(flat, self.b, self.b)
                X3 = K_blocks.reshape(flat, self.b, d)
                out3 = K_workspace.view(flat, self.b, d)
                torch.bmm(Sp_exp, X3, out=out3)
                if self.add_a:
                    # Sp@A is tiny ([1,H,NB,b,d]); one broadcast add kernel
                    SpA_k = torch.matmul(Sp_k_batch, A_k_batch)
                    out3.add_(SpA_k.expand(
                        batch_size, self.num_heads, num_blocks_, self.b, d
                    ).reshape(flat, self.b, d))
                K_blocks = K_workspace  # result buffer (out must not alias input)

            # --- Value Obfuscation ---
            if not self.fused:
                torch.matmul(V_blocks, M_v_batch, out=V_workspace)
                if rem > 0:
                    pad_ws = V_workspace.view(
                        batch_size, self.num_heads, num_blocks, self.b, d
                    )[:, :, -1, rem:, :]
                    pad_ws.uniform_(1.1, 1.5)
                    pad_ws.mul_(thresholds_v.to(V_workspace.dtype))
                if self.add_a:
                    V_workspace.add_(A_v_batch)
                Sp_v_batch = torch.matmul(S_v_batch, perm_matrices)
                torch.matmul(Sp_v_batch, V_workspace, out=V_blocks)
            else:
                Sp_v_batch = torch.matmul(S_v_batch, perm_matrices)
                Sp_exp_v = Sp_v_batch.expand(
                    batch_size, self.num_heads, num_blocks_, self.b, self.b
                ).reshape(flat, self.b, self.b)
                V3 = V_blocks.reshape(flat, self.b, d)
                out3v = V_workspace.view(flat, self.b, d)
                torch.bmm(Sp_exp_v, V3, out=out3v)
                if self.add_a:
                    SpA_v = torch.matmul(Sp_v_batch, A_v_batch)
                    out3v.add_(SpA_v.expand(
                        batch_size, self.num_heads, num_blocks_, self.b, d
                    ).reshape(flat, self.b, d))
                V_blocks = V_workspace

            K_kvcloak = K_blocks.view(batch_size, self.num_heads, padded_seq_len, d).to(
                K_dtype
            )
            V_kvcloak = V_blocks.view(batch_size, self.num_heads, padded_seq_len, d).to(
                K_dtype
            )

            kvcloaked_layers.append((K_kvcloak, V_kvcloak))

        return DynamicCache.from_legacy_cache(past_key_values=tuple(kvcloaked_layers))

    def deobfuscate(self, protected_kvcache: DynamicCache) -> DynamicCache:
        device = protected_kvcache.key_cache[0].device
        self._prepare_device_tensors(device)
        cached_inv_tensors = self._device_cache[device]["inv"]
        inversed_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, (K_kvcloak, V_kvcloak) in enumerate(protected_kvcache):
            batch_size, _, seq_len, d = K_kvcloak.shape
            num_blocks = seq_len // self.b
            K_dtype = K_kvcloak.dtype

            layer_tensors = cached_inv_tensors[layer_idx]
            inverse_S_k_batch = layer_tensors["inverse_S_k_batch"]
            inverse_M_k_batch = layer_tensors["inverse_M_k_batch"]
            inverse_S_v_batch = layer_tensors["inverse_S_v_batch"]
            inverse_M_v_batch = layer_tensors["inverse_M_v_batch"]

            thresholds_k = layer_tensors["thresholds_k"].view(
                1, self.num_heads, 1, 1, 1
            )
            thresholds_v = layer_tensors["thresholds_v"].view(
                1, self.num_heads, 1, 1, 1
            )

            # (num_heads, b+1, d) staggered marker lookup tables
            a_k_table = layer_tensors["a_k_list"]
            a_v_table = layer_tensors["a_v_list"]

            # No clone needed: the first matmul writes to K_workspace/V_workspace,
            # and the second writes back to K_blocks only after the workspace is
            # fully computed (no aliasing between matmul inputs and out).
            K_blocks = K_kvcloak.view(
                batch_size, self.num_heads, num_blocks, self.b, d
            ).to(self.dtype)
            V_blocks = V_kvcloak.view(
                batch_size, self.num_heads, num_blocks, self.b, d
            ).to(self.dtype)

            K_workspace = torch.empty_like(K_blocks)
            V_workspace = torch.empty_like(V_blocks)

            torch.matmul(inverse_S_k_batch, K_blocks, out=K_workspace)

            if self.add_a:
                self._strip_staggered_markers_(
                    K_workspace, a_k_table, layer_tensors["thresholds_k"]
                )

            if not self.fused:
                # padding lives in the kM domain (filled after the M
                # rotation): remove it BEFORE inverting M, then apply the
                # inverse rotation on the (non-block-aligned) remainder.
                K_ws = self._remove_padding_rows(
                    K_workspace.view(batch_size, self.num_heads, seq_len, d),
                    layer_tensors["thresholds_k"],
                    self.b,
                )
                K_blocks = torch.einsum(
                    "bhld,hde->bhle",
                    K_ws,
                    inverse_M_k_batch.view(self.num_heads, d, d),
                )
            else:
                # fused: the workspace already holds the final result; skip the copy
                K_blocks = K_workspace
            K_blocks = K_blocks.to(K_dtype)

            torch.matmul(inverse_S_v_batch, V_blocks, out=V_workspace)

            if self.add_a:
                self._strip_staggered_markers_(
                    V_workspace, a_v_table, layer_tensors["thresholds_v"]
                )

            if not self.fused:
                V_ws = self._remove_padding_rows(
                    V_workspace.view(batch_size, self.num_heads, seq_len, d),
                    layer_tensors["thresholds_v"],
                    self.b,
                )
                V_blocks = torch.einsum(
                    "bhld,hde->bhle",
                    V_ws,
                    inverse_M_v_batch.view(self.num_heads, d, d),
                )
            else:
                V_blocks = V_workspace
            V_blocks = V_blocks.to(K_dtype)

            if not self.fused:
                # padding was already removed before the inverse M rotation
                K_unpadded = K_blocks
                V_unpadded = V_blocks
            else:
                K_unpadded = K_blocks.view(batch_size, self.num_heads, seq_len, d)
                K_unpadded = self._remove_padding_rows(
                    K_unpadded, thresholds_k, self.b
                )
                V_unpadded = V_blocks.view(batch_size, self.num_heads, seq_len, d)
                V_unpadded = self._remove_padding_rows(
                    V_unpadded, thresholds_v, self.b
                )

            inversed_layers.append((K_unpadded, V_unpadded))

        return DynamicCache.from_legacy_cache(past_key_values=tuple(inversed_layers))


# ==============================================================================
# Test Setup and Execution
# ==============================================================================


def random_orthogonal_matrix(dim: int) -> torch.Tensor:
    random_matrix = torch.randn(dim, dim)
    S, _ = torch.linalg.qr(random_matrix)
    return S


def create_test_kv_config(
    num_layers, num_heads, theta, S_ratio, M_ratio, b, d, device, dtype
):
    """Generate test key configuration"""
    from defense.config.get_kvcloak_config import generate_marker_table

    threshold = [theta_ * M_ratio * 1.42 for theta_ in theta]
    kvcloak_config = []
    for _ in range(num_layers):
        layer_conf = []
        for _ in range(num_heads):
            head_conf = []
            for i in range(2):  # For K and V
                S = random_orthogonal_matrix(b)
                conf = {
                    "theta": theta[i],
                    "M_ratio": M_ratio,
                    "S_ratios": torch.rand(b) * (S_ratio - (1 / S_ratio))
                    + (1 / S_ratio),
                    "S": S,
                    "M_ratios": torch.rand(d // 2) * (M_ratio - (1 / M_ratio))
                    + (1 / M_ratio),
                    "M_angles": torch.rand(d // 2) * 2 * torch.pi,
                    "A": generate_marker_table(b, d, threshold[i]),
                    # -threshold[i] < kM < threshold[i]
                    # 3j*threshold[i] <= A[j] < (3j+0.5)*threshold[i]
                    # (3j-1)*threshold[i] < kM+a_j < (3j+2)*threshold[i]
                    # padding ~ U(1.1, 1.5)*threshold[i] (kM < padding < kM+a_j)
                }
                head_conf.append(conf)
            layer_conf.append(head_conf)
        kvcloak_config.append(layer_conf)
    return kvcloak_config


def create_test_kv_cache(batch_size, num_heads, seq_len, d, num_layers, device, dtype):
    """Create test KV Cache with values equal to their sequence indices."""
    past_key_values = []
    for _ in range(num_layers):
        # Create K, V tensors
        k_tensor = torch.zeros(
            batch_size, num_heads, seq_len, d, device=device, dtype=dtype
        )
        v_tensor = torch.zeros(
            batch_size, num_heads, seq_len, d, device=device, dtype=dtype
        )
        # Fill each sequence position with its index value
        for i in range(seq_len):
            k_tensor[:, :, i, :] = float(i) - seq_len / 2
            v_tensor[:, :, i, :] = float(i) - seq_len / 2
        past_key_values.append((k_tensor, v_tensor))
    return DynamicCache.from_legacy_cache(past_key_values=tuple(past_key_values))


def test():
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.manual_seed(42)
    # --- Test Parameters ---
    BATCH_SIZE = 1
    NUM_LAYERS = 2
    NUM_HEADS = 2
    SEQ_LEN = 20  # Not a multiple of b, to test padding logic
    S_RATIO = 1
    M_RATIO = 1
    THETA = [SEQ_LEN, SEQ_LEN]
    HEAD_DIM = 2  # Must be even
    BLOCK_SIZE = 16  # Block size b
    DEVICE = "cuda:0"
    DTYPE = torch.float32
    KVCLOAK_DTYPE = DTYPE
    fused = True
    # fused = False
    # need_ratio = True
    need_ratio = False
    # add_a = True
    add_a = False

    print(f"Test Environment: device={DEVICE}, dtype={DTYPE}")
    print(
        f"Parameters: BATCH={BATCH_SIZE}, LAYERS={NUM_LAYERS}, HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}, BLOCK_SIZE={BLOCK_SIZE}\n"
    )

    # 1. Create test keys and original KV Cache
    kv_config = create_test_kv_config(
        NUM_LAYERS,
        NUM_HEADS,
        THETA,
        S_RATIO,
        M_RATIO,
        BLOCK_SIZE,
        HEAD_DIM,
        DEVICE,
        DTYPE,
    )
    kvcloak = KVCloak(kv_config, KVCLOAK_DTYPE, fused, need_ratio, add_a)
    original_kvcache = create_test_kv_cache(
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, NUM_LAYERS, DEVICE, DTYPE
    )

    # Extract original K tensor for comparison
    original_k, _ = original_kvcache[0]
    # _, original_k = original_kvcache[0]
    original_k_head0 = original_k[
        0, 0, :, :
    ]  # Get batch 0, head 0, all seq, dim 0

    # 2. Execute obfuscation
    print("--- Step 1: Execute kvcloak_obf for obfuscation ---")
    protected_kvcache = kvcloak.obfuscate(original_kvcache)
    print("Obfuscation complete.\n")

    # 3. Execute deobfuscation
    print("--- Step 2: Execute kvcloak_inv for deobfuscation ---")
    inversed_kvcache = kvcloak.deobfuscate(protected_kvcache)
    print("Deobfuscation complete.\n")

    # 4. Result verification
    print("--- Step 3: Verify results ---")
    inversed_k, _ = inversed_kvcache[0]
    # _, inversed_k = inversed_kvcache[0]
    inversed_k_head0 = inversed_k[0, 0, :, 0]

    # Get permutation matrix P to analyze errors
    print("Original K tensor (values of first head):")
    print(original_k_head0)
    print("\nDeobfuscated K tensor (values of first head):")
    print(inversed_k_head0)

    # Check the order of the first block (0-15)
    original_block1_vals = set(
        original_k_head0[:BLOCK_SIZE].to(torch.float32).cpu().numpy()
    )
    inversed_block1_vals = set(
        torch.round(inversed_k_head0[:BLOCK_SIZE]).to(torch.float32).cpu().numpy()
    )

    print("\n--- Analysis ---")
    if original_k.shape != inversed_k.shape:
        print(
            f"❌ Fail: Shape mismatch! Original shape {original_k.shape}, restored shape {inversed_k.shape}"
        )
    else:
        print(
            f"✅ Pass: Shapes match. Original shape {original_k.shape}, restored shape {inversed_k.shape}"
        )

    if torch.allclose(original_k_head0, inversed_k_head0):
        print("❌ Pass: Tensor content and order completely identical. This could be an extremely rare event.")
    else:
        if original_block1_vals == inversed_block1_vals:
            print("✅ Observation: Values in the first block are the same set, but order is shuffled.")
        else:
            print("❌ Observation: Values themselves were not correctly restored.")


def process_model_cache(
    model_name: str,
    cache_path: str,
    kvcloak: KVCloak,
    device: torch.device,
    protect_type: str = "kvcloak",
    cloak_factory=None,
):
    """Protect stored KV caches.

    Legacy behaviour (cloak_factory=None) reuses the single configured
    `kvcloak` object across every sample directory. Sharing one parameter
    instance across requests is acceptable ONLY for reproducing published
    benchmarks; deployments must derive parameters per session.

    When cloak_factory is given (see defense.core.keyderive.
    session_cloak_factory), each sample is protected under a freshly
    derived per-session parameter instance and the public session nonce
    is stored next to the protected cache so the receiver can re-derive
    it. This is the required deployment mode.
    """
    if not cache_path.exists():
        print(f"Warning: KV cache directory not found for {model_name} at {cache_path}")
        return

    # Use glob to get subdirectories
    sub_directories = [item for item in cache_path.glob("*") if item.is_dir()]

    if not sub_directories:
        print(f"No subdirectories found in '{cache_path}' to process.")
        return

    if cloak_factory is None:
        print(
            "WARNING: protecting multiple caches under ONE static configuration "
            "(benchmark/research mode). Deployments must derive parameters per "
            "session: pass --user-key (see defense/core/keyderive.py)."
        )

    print(f"Processing directories in '{cache_path}':")
    for item_dir in tqdm(sub_directories, desc=f"Processing {model_name}"):
        input_hash = item_dir.name
        orig_kvcache_path = item_dir / "origin" / "past_key_values.pt"

        if not orig_kvcache_path.exists():
            print(f"Warning: Original KV cache not found at {orig_kvcache_path}")
            continue

        try:
            orig_kvcache = torch.load(orig_kvcache_path, weights_only=True)
        except Exception as e:
            print(
                f"Error loading original KV cache for {input_hash} in {model_name}: {e}"
            )
            continue

        session_nonce = None
        if cloak_factory is not None:
            kvcloak, session_nonce = cloak_factory()
            dtype_for_load = kvcloak.dtype
        else:
            dtype_for_load = kvcloak.dtype

        orig_kvcache = tuple(
            (
                k.to(device=device, dtype=dtype_for_load),
                v.to(device=device, dtype=dtype_for_load),
            )
            for (k, v) in orig_kvcache
        )

        # Process and save KV Cloak cache
        kvcloak_kvcache = kvcloak.obfuscate(orig_kvcache)
        kvcloak_kvcache = tuple(
            (k.to(device="cpu"), v.to(device="cpu")) for (k, v) in kvcloak_kvcache
        )
        kvcloak_output_dir = item_dir / protect_type
        kvcloak_output_dir.mkdir(parents=True, exist_ok=True)
        kvcloak_kvcache_path = kvcloak_output_dir / "past_key_values.pt"
        torch.save(kvcloak_kvcache, kvcloak_kvcache_path)
        if session_nonce is not None:
            # Public metadata: the receiver re-derives (S, M, A) from the
            # user key and this nonce. Without the user key it is useless.
            (kvcloak_output_dir / "session_nonce.bin").write_bytes(session_nonce)


def main():
    parser = argparse.ArgumentParser(description="Apply KV-Cloak to KV-cache files.")
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--dataset-path",
        default="./dataset/lmsys-chat-1m_1k.jsonl",
        help="Dataset path used to infer cache directory name.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        # float32 default: bfloat16 storage is incompatible with the staggered
        # marker amplitudes (rounding at 2^-9 of marker scale corrupts decoded
        # values and breaks padding-row classification). Use bf16 only with an
        # explicit flag until the low-precision design is settled.
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--theta-ratio", type=float, default=2.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--S-ratio", type=float, default=1.0)
    parser.add_argument("--M-ratio", type=float, default=1.0)
    parser.add_argument(
        "--fused",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether fusion has already absorbed M matrix transforms.",
    )
    parser.add_argument(
        "--need-ratio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--add-a",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to add the staggered per-row markers (the paper design).",
    )
    parser.add_argument(
        "--protect-type",
        default="kvcloak",
        help="Output subdirectory name under each sample directory.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Direct cache directory path. If not set, built from dataset/model/dtype.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="KV-Cloak config path (.pt). If not set, built from block/S/M/theta/model.",
    )
    parser.add_argument(
        "--user-key",
        default=None,
        help=(
            "Hex-encoded user secret key. When set, each cache file is protected "
            "under its own per-session derived (S, M, A) instance (secure mode, "
            "required for deployment). Requires --theta-config and --head-dim."
        ),
    )
    parser.add_argument(
        "--theta-config",
        default=None,
        help="Public theta calibration JSON (same format as get_kvcloak_config).",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Attention head dim (required in --user-key mode; layers/kv-heads are "
        "inferred from the theta config).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.serialization.add_safe_globals([DynamicCache, set])

    dtype = getattr(torch, args.dtype)
    dtype_name = args.dtype
    dataset_name = Path(args.dataset_path).stem

    cache_path = (
        Path(args.cache_path).expanduser()
        if args.cache_path is not None
        else Path(
            f"cache/{dtype_name}/{dataset_name}/{args.model_name}"
        ).expanduser()
    )

    if args.user_key is not None:
        # Secure mode: one derived parameter instance per cache file.
        import json as _json
        import sys as _sys

        _repo_root = str(Path(__file__).resolve().parents[2])
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)

        from defense.core.keyderive import session_cloak_factory

        theta_config_path = (
            Path(args.theta_config).expanduser()
            if args.theta_config is not None
            else Path(
                f"defense/config/kvcloak/theta/{args.model_name}.json"
            ).expanduser()
        )
        if not theta_config_path.exists():
            raise FileNotFoundError(
                f"theta config not found: {theta_config_path}"
            )
        with theta_config_path.open("r", encoding="utf-8") as f:
            theta_config = _json.load(f)
        if args.head_dim is None:
            raise ValueError("--head-dim is required in --user-key mode")
        user_key = bytes.fromhex(args.user_key)
        if not user_key:
            raise ValueError("--user-key must be a non-empty hex string")
        cloak_factory = session_cloak_factory(
            user_key=user_key,
            theta_config=theta_config,
            head_dim=args.head_dim,
            block_size=args.block_size,
            dtype=dtype,
            fused=args.fused,
            need_ratio=args.need_ratio,
            add_a=args.add_a,
            S_ratio=args.S_ratio,
            M_ratio=args.M_ratio,
            theta_ratio=args.theta_ratio,
        )
        print(
            f"Configuration - {args.model_name}_{dtype_name}_a{args.add_a}_f{args.fused}"
            f"_b{args.block_size}_S{args.S_ratio:g}_M{args.M_ratio:g}_t{args.theta_ratio:g}"
            f" {args.device} [PER-SESSION DERIVATION]"
        )
        process_model_cache(
            args.model_name,
            cache_path,
            None,
            args.device,
            protect_type=args.protect_type,
            cloak_factory=cloak_factory,
        )
        return

    kvcloak_config_path = (
        Path(args.config_path).expanduser()
        if args.config_path is not None
        else Path(
            f"defense/config/kvcloak/b{args.block_size}_S{args.S_ratio:g}_M{args.M_ratio:g}_t{args.theta_ratio:g}/{args.model_name}.pt"
        ).expanduser()
    )
    if not kvcloak_config_path.exists():
        raise FileNotFoundError(
            f"KV-Cloak config path does not exist: {kvcloak_config_path}"
        )

    cache_path = (
        Path(args.cache_path).expanduser()
        if args.cache_path is not None
        else Path(
            f"cache/{dtype_name}/{dataset_name}/{args.model_name}"
        ).expanduser()
    )
    print(
        f"Configuration - {args.model_name}_{dtype_name}_a{args.add_a}_f{args.fused}_b{args.block_size}_S{args.S_ratio:g}_M{args.M_ratio:g}_t{args.theta_ratio:g} {args.device}"
    )

    kvcloak_config = torch.load(kvcloak_config_path)
    kvcloak = KVCloak(kvcloak_config, dtype, args.fused, args.need_ratio, args.add_a)
    process_model_cache(
        args.model_name,
        cache_path,
        kvcloak,
        args.device,
        protect_type=args.protect_type,
    )


if __name__ == "__main__":
    main()
