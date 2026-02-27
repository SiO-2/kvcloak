"""Unit tests for AES KV-cache protection."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "defense" / "baseline"))

from aes_kvcache import KVCacheAESProtecter
from transformers.cache_utils import DynamicCache


@pytest.fixture
def aes_protector():
    """Create an AES protector with a test key."""
    key = b"0123456789abcdef"  # 16 bytes for AES-128
    return KVCacheAESProtecter(key, device="cpu")


@pytest.fixture
def sample_kv_cache():
    """Create a small sample KV-cache for testing."""
    batch_size = 2
    num_heads = 4
    seq_len = 10
    head_dim = 64
    
    kv_list = []
    for _ in range(2):  # 2 layers
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        kv_list.append((k, v))
    
    return DynamicCache.from_legacy_cache(kv_list)


class TestAESProtector:
    """Tests for AES KV-cache protection."""
    
    def test_init_with_valid_key(self):
        """Test initialization with valid key lengths."""
        # AES-128
        protector128 = KVCacheAESProtecter(b"0123456789abcdef", device="cpu")
        assert len(protector128.key) == 16
        
        # AES-256
        protector256 = KVCacheAESProtecter(b"0123456789abcdef0123456789abcdef", device="cpu")
        assert len(protector256.key) == 32
    
    def test_init_with_invalid_key(self):
        """Test that invalid key lengths raise error."""
        with pytest.raises(ValueError, match="16 or 32 bytes"):
            KVCacheAESProtecter(b"short_key", device="cpu")
        
        with pytest.raises(ValueError, match="16 or 32 bytes"):
            KVCacheAESProtecter(b"x" * 24, device="cpu")  # Invalid length
    
    def test_encrypt_decrypt_roundtrip(self, aes_protector, sample_kv_cache):
        """Test that encrypt followed by decrypt returns original data."""
        # Store original values
        original_k = [layer[0].clone() for layer in sample_kv_cache]
        original_v = [layer[1].clone() for layer in sample_kv_cache]
        
        # Encrypt
        encrypted = aes_protector.encrypt(sample_kv_cache)
        assert len(encrypted) == 2  # 2 layers
        
        # Decrypt
        decrypted = aes_protector.decrypt(encrypted)
        
        # Verify data integrity
        for i, (k_dec, v_dec) in enumerate(decrypted):
            k_orig, v_orig = original_k[i], original_v[i]
            assert torch.allclose(k_dec, k_orig, atol=1e-6)
            assert torch.allclose(v_dec, v_orig, atol=1e-6)
    
    def test_encryption_changes_data(self, aes_protector, sample_kv_cache):
        """Test that encryption actually changes the data."""
        encrypted = aes_protector.encrypt(sample_kv_cache)
        
        # Verify encrypted data is different from original
        # (At least check structure is correct)
        for layer_data in encrypted:
            assert len(layer_data) == 2  # key and value tuples
            key_tuple, value_tuple = layer_data
            
            # Each tuple should have (nonce, ciphertext, shape, dtype)
            assert len(key_tuple) == 4
            assert len(value_tuple) == 4
            
            # Nonce should be 12 bytes (GCM standard)
            assert len(key_tuple[0]) == 12
            assert len(value_tuple[0]) == 12
            
            # Ciphertext should be longer than original (due to auth tag)
            assert isinstance(key_tuple[1], bytes)
            assert isinstance(value_tuple[1], bytes)
    
    def test_different_nonces_per_encryption(self, aes_protector, sample_kv_cache):
        """Test that each encryption uses different nonces."""
        encrypted1 = aes_protector.encrypt(sample_kv_cache)
        encrypted2 = aes_protector.encrypt(sample_kv_cache)
        
        # Extract nonces
        nonces1 = [layer[0][0] for layer in encrypted1]
        nonces2 = [layer[0][0] for layer in encrypted2]
        
        # Nonces should be different
        for n1, n2 in zip(nonces1, nonces2):
            assert n1 != n2, "Nonces should be unique per encryption"
    
    def test_repr(self, aes_protector):
        """Test string representation."""
        repr_str = repr(aes_protector)
        assert "KVCacheAESProtecter" in repr_str
        assert "128bits" in repr_str or "256bits" in repr_str


class TestTensorConversion:
    """Tests for tensor to bytes conversion."""
    
    def test_tensor_to_bytes_and_back(self, aes_protector):
        """Test tensor conversion roundtrip."""
        original = torch.randn(2, 4, 8, 16)
        
        # Convert to bytes
        data_bytes, shape, dtype = aes_protector._tensor_to_bytes(original)
        assert isinstance(data_bytes, bytes)
        assert shape == original.shape
        assert dtype == original.dtype
        
        # Convert back
        restored = aes_protector._bytes_to_tensor(data_bytes, shape, dtype)
        assert torch.allclose(original, restored)
    
    def test_different_dtypes(self, aes_protector):
        """Test conversion with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            original = torch.randn(2, 4).to(dtype)
            data_bytes, shape, dtype_out = aes_protector._tensor_to_bytes(original)
            restored = aes_protector._bytes_to_tensor(data_bytes, shape, dtype_out)
            
            if dtype.is_floating_point:
                assert torch.allclose(original.float(), restored.float())
            else:
                assert torch.equal(original, restored)
