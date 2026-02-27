"""Unit tests for security utilities."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from security_utils import (
    validate_path,
    validate_model_name,
    safe_join,
    PathSecurityError,
)


class TestValidatePath:
    """Tests for validate_path function."""
    
    def test_valid_relative_path(self, tmp_path):
        """Test that valid relative paths pass."""
        result = validate_path("test/file.txt", base_dir=tmp_path)
        assert result == tmp_path / "test/file.txt"
    
    def test_path_with_parent_traversal(self):
        """Test that paths with '..' are rejected."""
        with pytest.raises(PathSecurityError, match="parent directory"):
            validate_path("../etc/passwd")
    
    def test_path_with_null_bytes(self):
        """Test that paths with null bytes are rejected."""
        with pytest.raises(PathSecurityError, match="null bytes"):
            validate_path("file\x00.txt")
    
    def test_absolute_path_not_allowed(self):
        """Test that absolute paths are rejected by default."""
        with pytest.raises(PathSecurityError, match="Absolute paths"):
            validate_path("/etc/passwd")
    
    def test_absolute_path_allowed(self):
        """Test that absolute paths work when allowed."""
        result = validate_path("/tmp/test", allow_absolute=True)
        assert result == Path("/tmp/test")
    
    def test_must_exist_validation(self, tmp_path):
        """Test must_exist parameter."""
        # Create a file
        test_file = tmp_path / "exists.txt"
        test_file.write_text("test")
        
        # Should pass for existing file (with allow_absolute since tmp_path is absolute)
        result = validate_path(test_file, must_exist=True, allow_absolute=True)
        assert result.exists()
        
        # Should fail for non-existing file
        with pytest.raises(PathSecurityError, match="does not exist"):
            validate_path(tmp_path / "not_exists.txt", must_exist=True, allow_absolute=True)
    
    def test_base_directory_constraint(self, tmp_path):
        """Test that paths must be within base directory."""
        # Valid path within base
        result = validate_path("subdir/file.txt", base_dir=tmp_path)
        assert str(result).startswith(str(tmp_path))
        
        # Invalid path outside base
        with pytest.raises(PathSecurityError, match="not within"):
            validate_path("/outside/path", base_dir=tmp_path)


class TestValidateModelName:
    """Tests for validate_model_name function."""
    
    @pytest.mark.parametrize("name", [
        "Llama-3.2-1B",
        "gpt2",
        "Meta-Llama-3.1-8B",
        "model_v1.0",
        "test_model",
    ])
    def test_valid_model_names(self, name):
        """Test various valid model names."""
        result = validate_model_name(name)
        assert result == name
    
    @pytest.mark.parametrize("name", [
        "model/../other",
        "model;rm -rf",
        "model\x00hidden",
        "model$(whoami)",
        "",
    ])
    def test_invalid_model_names(self, name):
        """Test that invalid names raise exception."""
        with pytest.raises(PathSecurityError, match="unsafe characters"):
            validate_model_name(name)


class TestSafeJoin:
    """Tests for safe_join function."""
    
    def test_safe_join_basic(self, tmp_path):
        """Test basic path joining."""
        result = safe_join(tmp_path, "a", "b", "c.txt")
        assert result == tmp_path / "a" / "b" / "c.txt"
    
    def test_safe_join_prevents_escape(self, tmp_path):
        """Test that safe_join prevents directory escape."""
        with pytest.raises(PathSecurityError, match="escapes base"):
            safe_join(tmp_path, "..", "etc", "passwd")
    
    def test_safe_join_with_subdirectories(self, tmp_path):
        """Test joining with multiple subdirectories."""
        result = safe_join(tmp_path, "level1", "level2", "file.txt")
        assert result == tmp_path / "level1" / "level2" / "file.txt"
        # Verify it's still within base
        assert str(result).startswith(str(tmp_path))
