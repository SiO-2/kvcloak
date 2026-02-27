# aes_kvcache.py
"""AES-GCM encryption for KV-cache protection."""

import os
from typing import Tuple, Union
import torch
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from transformers.cache_utils import DynamicCache


class KVCacheAESProtecter:
    """AES-GCM encryption protector for KV-Cache.
    
    Encrypts and decrypts KV-Cache tensors using AES-GCM mode.
    Each tensor is encrypted with a unique nonce for security.
    
    Attributes:
        key: AES key (16 or 32 bytes)
        device: Target device for tensors
        nonce_size: Size of GCM nonce (12 bytes)
    """

    def __init__(self, key: bytes, device: Union[str, torch.device] = "cpu"):
        """Initialize AES protector.

        Args:
            key: AES key (16 or 32 bytes for AES-128/256)
            device: Device for output tensors

        Raises:
            ValueError: If key length is not 16 or 32 bytes
        """
        self.nonce_size = 12  # GCM standard nonce size
        if len(key) not in [16, 32]:
            raise ValueError("AES key must be 16 or 32 bytes (AES-128 or AES-256)")
        self.key = key
        self.device = device

    def _tensor_to_bytes(
        self, tensor: torch.Tensor
    ) -> "Tuple[bytes, Tuple[int, ...], torch.dtype]":
        """Convert tensor to bytes while preserving metadata.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (bytes_data, shape, dtype)
        """
        tensor_np = tensor.cpu().numpy()
        return tensor_np.tobytes(), tensor_np.shape, tensor.dtype

    def _bytes_to_tensor(
        self, data: bytes, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert bytes back to tensor.
        
        Args:
            data: Byte data
            shape: Original tensor shape
            dtype: Original tensor dtype
            
        Returns:
            Reconstructed tensor
        """
        # Map torch dtype to numpy dtype
        dtype_str = str(dtype).split(".")[-1]
        np_dtype = np.dtype(dtype_str)
        tensor_np = np.frombuffer(data, dtype=np_dtype).copy()
        return torch.from_numpy(tensor_np).reshape(shape).to(self.device)

    def encrypt(self, past_key_values: DynamicCache) -> list:
        """Encrypt the entire KV-Cache.

        Args:
            past_key_values: KV cache to encrypt

        Returns:
            List of encrypted layers, each containing [key_tuple, value_tuple]
            where tuple is (nonce, ciphertext, shape, dtype)
        """
        encrypted_cache = []
        num_layers = len(past_key_values.key_cache)
        
        for i in range(num_layers):
            key_states = past_key_values.key_cache[i]
            value_states = past_key_values.value_cache[i]
            aesgcm = AESGCM(self.key)

            # Encrypt Key
            key_bytes, key_shape, key_dtype = self._tensor_to_bytes(key_states)
            nonce_k = os.urandom(self.nonce_size)
            ct_bytes_k = aesgcm.encrypt(nonce_k, key_bytes, None)
            encrypted_key = (nonce_k, ct_bytes_k, key_shape, key_dtype)

            # Encrypt Value
            value_bytes, value_shape, value_dtype = self._tensor_to_bytes(value_states)
            nonce_v = os.urandom(self.nonce_size)
            ct_bytes_v = aesgcm.encrypt(nonce_v, value_bytes, None)
            encrypted_value = (nonce_v, ct_bytes_v, value_shape, value_dtype)

            encrypted_cache.append([encrypted_key, encrypted_value])

        return encrypted_cache

    def decrypt(self, encrypted_cache: list) -> DynamicCache:
        """Decrypt the entire KV-Cache.

        Args:
            encrypted_cache: Encrypted cache from encrypt()

        Returns:
            Decrypted DynamicCache
        """
        decrypted_kv_list = []
        
        for encrypted_key, encrypted_value in encrypted_cache:
            aesgcm = AESGCM(self.key)

            # Decrypt Key
            nonce_k, ct_bytes_k, key_shape, key_dtype = encrypted_key
            pt_bytes_k = aesgcm.decrypt(nonce_k, ct_bytes_k, None)
            key_states = self._bytes_to_tensor(pt_bytes_k, key_shape, key_dtype)

            # Decrypt Value
            nonce_v, ct_bytes_v, value_shape, value_dtype = encrypted_value
            pt_bytes_v = aesgcm.decrypt(nonce_v, ct_bytes_v, None)
            value_states = self._bytes_to_tensor(pt_bytes_v, value_shape, value_dtype)

            decrypted_kv_list.append((key_states, value_states))

        return DynamicCache.from_legacy_cache(decrypted_kv_list)

    def __repr__(self) -> str:
        """String representation."""
        return f"KVCacheAESProtecter(key_len={len(self.key)*8}bits, device={self.device})"
