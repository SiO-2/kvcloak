"""Unit tests for configuration module."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    MODEL_CONFIGS,
    KVCLOAK_DEFAULTS,
    DP_DEFAULTS,
    ATTACK_DEFAULTS,
    BITTER_LESSON_TEXT,
    get_model_path,
    validate_model_name,
    get_cache_path,
)


class TestModelConfigs:
    """Tests for model configurations."""
    
    def test_model_configs_exist(self):
        """Test that model configs are defined."""
        assert len(MODEL_CONFIGS) > 0
        assert "Llama-3.2-1B" in MODEL_CONFIGS
        assert "gpt2" in MODEL_CONFIGS
    
    def test_model_config_structure(self):
        """Test that each model config has correct structure."""
        for model_name, config in MODEL_CONFIGS.items():
            assert len(config) == 4, f"Model {model_name} should have 4 config values"
            # [base_model, logical_batch, micro_batch, target_gap]
            base_model, lb, mb, gap = config
            assert isinstance(base_model, str)
            assert isinstance(lb, int) and lb > 0
            assert isinstance(mb, int) and mb > 0
            assert isinstance(gap, int)
    
    def test_validate_model_name_valid(self):
        """Test validation of known models."""
        for model in MODEL_CONFIGS:
            assert validate_model_name(model) == model
    
    def test_validate_model_name_invalid(self):
        """Test validation of unknown models."""
        with pytest.raises(ValueError, match="Unknown model"):
            validate_model_name("nonexistent-model")


class TestKVCacheConfig:
    """Tests for KV-cache configuration."""
    
    def test_kvcloak_defaults_structure(self):
        """Test KV-cloak defaults have required keys."""
        required_keys = [
            "block_size", "S_ratio", "M_ratio", 
            "theta_ratio", "fuse", "add_a", "need_ratio"
        ]
        for key in required_keys:
            assert key in KVCLOAK_DEFAULTS
    
    def test_kvcloak_default_values(self):
        """Test KV-cloak default values are reasonable."""
        assert KVCLOAK_DEFAULTS["block_size"] > 0
        assert KVCLOAK_DEFAULTS["S_ratio"] > 0
        assert KVCLOAK_DEFAULTS["M_ratio"] > 0
        assert KVCLOAK_DEFAULTS["theta_ratio"] > 0
        assert isinstance(KVCLOAK_DEFAULTS["fuse"], bool)


class TestDPConfig:
    """Tests for DP configuration."""
    
    def test_dp_defaults_structure(self):
        """Test DP defaults have required keys."""
        required_keys = ["norm_percentile", "epsilon", "delta"]
        for key in required_keys:
            assert key in DP_DEFAULTS
    
    def test_dp_default_values(self):
        """Test DP default values are reasonable."""
        assert 0 <= DP_DEFAULTS["norm_percentile"] <= 100
        assert DP_DEFAULTS["epsilon"] > 0
        assert 0 < DP_DEFAULTS["delta"] < 1


class TestAttackConfig:
    """Tests for attack configuration."""
    
    def test_attack_defaults_structure(self):
        """Test attack defaults have required keys."""
        required_keys = [
            "logical_batch_size", "micro_batch_size", 
            "stop_partition", "target_gap", "enhance"
        ]
        for key in required_keys:
            assert key in ATTACK_DEFAULTS
    
    def test_bitter_lesson_text(self):
        """Test that bitter lesson text is defined and non-empty."""
        assert len(BITTER_LESSON_TEXT) > 0
        assert "bitter lesson" in BITTER_LESSON_TEXT.lower()


class TestPathUtilities:
    """Tests for path utility functions."""
    
    def test_get_model_path(self):
        """Test model path generation."""
        path = get_model_path("test-model")
        assert isinstance(path, Path)
        assert "test-model" in str(path)
    
    def test_get_cache_path(self):
        """Test cache path generation."""
        path = get_cache_path(
            model_name="Llama-3.2-1B",
            dataset_name="test-dataset",
            dtype=torch.float32,
            protect_type="kvcloak"
        )
        assert isinstance(path, Path)
        assert "Llama-3.2-1B" in str(path)
        assert "kvcloak" in str(path)
