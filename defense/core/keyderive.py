"""Per-session derivation of the KV-Cloak secret matrices (S, M, A).

Deployment requirement: parameter isolation
-------------------------------------------
KV-Cloak's confidentiality argument holds only when the secret matrices
are NOT shared across independent requests. The released research
prototype loads one static configuration file and reuses it for every
request, which is acceptable solely for reproducing published
benchmarks. Any deployment must derive (S, M, A) per user session:

    seed      = HKDF-SHA256(ikm = user_key, salt = session_nonce,
                            info = "kvcloak/session-config/v1" | geometry)
    (S, M, A) = get_kvcloak_config(...) with all randomness drawn from a
                torch.Generator seeded with `seed`

Properties
----------
* The session nonce is PUBLIC metadata stored next to the protected
  cache (the receiver needs it to re-derive the matrices); the user key
  never leaves the trust boundary (client / TEE).
* Determinism: the same (user_key, session_nonce, geometry) yields
  bitwise-identical matrices, so the obfuscating side and the
  deobfuscating side re-derive the same instance without exchanging any
  secret beyond the user key.
* Nonce uniqueness: new_session_nonce() draws 128 random bits. Under one
  key, two sessions share matrices only on a 64-bit DRBG-seed collision
  (negligible for realistic session counts), and an adversary who knows
  its own nonce cannot find a colliding victim nonce in feasible time.
* Cross-instance isolation is covered by the session-derivation tests:
  statistics learned under one session do not decode blocks protected
  under another.
"""

import hashlib
import hmac
import os
from typing import Callable, Dict, Optional, Tuple

import torch

from defense.config.get_kvcloak_config import get_kvcloak_config
from defense.core.kvcloak import KVCloak

INFO_PREFIX = b"kvcloak/session-config/v1"


def new_session_nonce(nbytes: int = 16) -> bytes:
    """Fresh public session nonce (128 random bits by default).

    Must never repeat under the same user key: a repeated nonce re-derives
    the same (S, M, A) instance and therefore re-opens the shared-
    configuration condition between the two sessions.
    """
    return os.urandom(nbytes)


def _hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869 HKDF with SHA-256, standard library only."""
    if not salt:
        salt = b"\x00" * hashlib.sha256().digest_size
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


def geometry_label(
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    block_size: int,
    S_ratio: float,
    M_ratio: float,
    theta_ratio: float,
) -> str:
    """Bind the deployment geometry into the HKDF info field so that the
    same (key, nonce) used with different geometry derives independent
    matrices instead of reusing low-dimensional components."""
    return (
        f"layers={num_hidden_layers};kv_heads={num_key_value_heads};"
        f"head_dim={head_dim};block={block_size};"
        f"S_ratio={S_ratio:g};M_ratio={M_ratio:g};theta_ratio={theta_ratio:g}"
    )


def session_seed(user_key: bytes, session_nonce: bytes, geometry: str) -> int:
    """Derive the per-session DRBG seed from the user key and nonce."""
    if not user_key:
        raise ValueError("user_key must be non-empty secret bytes")
    if not session_nonce:
        raise ValueError("session_nonce must be non-empty public bytes")
    okm = _hkdf_sha256(
        ikm=bytes(user_key),
        salt=bytes(session_nonce),
        info=INFO_PREFIX + b"|" + geometry.encode(),
        length=32,
    )
    return int.from_bytes(okm[:8], "big")


def derive_session_config(
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    block_size: int,
    theta_config: Dict,
    *,
    user_key: bytes,
    session_nonce: bytes,
    S_ratio: float = 1.0,
    M_ratio: float = 1.0,
    theta_ratio: float = 2.0,
):
    """Derive one session's KV-Cloak configuration from (user_key, nonce).

    theta_config is the public per-model calibration (layer/head value
    statistics) in the same format get_kvcloak_config expects; it is not
    secret material.
    """
    geometry = geometry_label(
        num_hidden_layers,
        num_key_value_heads,
        head_dim,
        block_size,
        S_ratio,
        M_ratio,
        theta_ratio,
    )
    seed = session_seed(user_key, session_nonce, geometry)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return get_kvcloak_config(
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        block_size=block_size,
        theta_config=theta_config,
        S_ratio=S_ratio,
        M_ratio=M_ratio,
        theta_ratio=theta_ratio,
        generator=generator,
    )


class SessionKVCloak(KVCloak):
    """A KV-Cloak instance bound to exactly one derived parameter instance.

    Sender side:   cloak = SessionKVCloak(user_key=k, session_nonce=n, ...);
                   store cloak.session_nonce next to the protected cache.
    Receiver side: rebuild with the same user_key and the stored (public)
                   session_nonce and call deobfuscate(); any other nonce
                   yields independent matrices and therefore only garbage.

    Creating a new SessionKVCloak with a fresh nonce -- rather than reusing
    an object or resampling the row permutation -- is what enforces the
    per-session parameter-isolation boundary.
    """

    def __init__(
        self,
        *,
        user_key: bytes,
        session_nonce: bytes,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        block_size: int,
        theta_config: Dict,
        dtype: torch.dtype = torch.float32,
        fused: bool = False,
        need_ratio: bool = False,
        add_a: bool = True,
        S_ratio: float = 1.0,
        M_ratio: float = 1.0,
        theta_ratio: float = 2.0,
    ):
        self.user_key = bytes(user_key)
        self.session_nonce = bytes(session_nonce)
        config = derive_session_config(
            num_hidden_layers,
            num_key_value_heads,
            head_dim,
            block_size,
            theta_config,
            user_key=self.user_key,
            session_nonce=self.session_nonce,
            S_ratio=S_ratio,
            M_ratio=M_ratio,
            theta_ratio=theta_ratio,
        )
        super().__init__(config, dtype, fused, need_ratio, add_a)


def session_cloak_factory(
    *,
    user_key: bytes,
    theta_config: Dict,
    num_hidden_layers: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype = torch.float32,
    fused: bool = False,
    need_ratio: bool = False,
    add_a: bool = True,
    S_ratio: float = 1.0,
    M_ratio: float = 1.0,
    theta_ratio: float = 2.0,
) -> Callable[[], Tuple[SessionKVCloak, bytes]]:
    """Return a zero-argument factory producing a fresh SessionKVCloak with
    an independent nonce per call. Layers/kv-heads default to the shape of
    theta_config when omitted."""

    if num_hidden_layers is None:
        num_hidden_layers = len(theta_config)
    if num_key_value_heads is None:
        num_key_value_heads = len(
            theta_config["layer_0"]["key_max_values"]
        )

    def factory() -> Tuple[SessionKVCloak, bytes]:
        nonce = new_session_nonce()
        cloak = SessionKVCloak(
            user_key=user_key,
            session_nonce=nonce,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            block_size=block_size,
            theta_config=theta_config,
            dtype=dtype,
            fused=fused,
            need_ratio=need_ratio,
            add_a=add_a,
            S_ratio=S_ratio,
            M_ratio=M_ratio,
            theta_ratio=theta_ratio,
        )
        return cloak, nonce

    return factory
