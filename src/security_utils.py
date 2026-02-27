"""
Security utilities for path validation and input sanitization.
"""

from pathlib import Path
from typing import Union, Optional
import re


class PathSecurityError(Exception):
    """Raised when a path fails security checks."""
    pass


def validate_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    must_exist: bool = False,
    allow_absolute: bool = False
) -> Path:
    """
    Validate a path for security issues.
    
    Args:
        path: The path to validate
        base_dir: Optional base directory that the path must be within
        must_exist: Whether the path must exist
        allow_absolute: Whether to allow absolute paths
        
    Returns:
        Validated Path object
        
    Raises:
        PathSecurityError: If path fails security checks
    """
    # Convert to string for initial checks
    path_str = str(path)
    
    # Check for null bytes first (before any path operations)
    if "\x00" in path_str:
        raise PathSecurityError("Path contains null bytes")
    
    # Check for path traversal attempts in the original string
    if ".." in path_str:
        raise PathSecurityError(f"Path contains parent directory reference: {path}")
    
    path = Path(path).expanduser()
    is_absolute = path.is_absolute()
    
    # If base_dir is provided, we allow absolute paths but check they're within base_dir
    if base_dir is not None:
        base_dir = Path(base_dir).expanduser().resolve()
        
        # If path is absolute, use it directly; otherwise join with base_dir
        if is_absolute:
            resolved_path = path.resolve()
        else:
            # Join with base_dir and resolve
            full_path = base_dir / path
            resolved_path = full_path.resolve()
        
        try:
            resolved_path.relative_to(base_dir)
        except ValueError:
            raise PathSecurityError(
                f"Path {resolved_path} is not within allowed directory {base_dir}"
            )
        
        # Check existence on resolved path
        if must_exist and not resolved_path.exists():
            raise PathSecurityError(f"Path does not exist: {resolved_path}")
        
        return resolved_path
    
    # No base_dir - check absolute paths
    if is_absolute and not allow_absolute:
        raise PathSecurityError(f"Absolute paths not allowed: {path}")
    
    # Check existence if requested (on non-resolved path)
    if must_exist and not path.exists():
        raise PathSecurityError(f"Path does not exist: {path}")
    
    return path


def validate_model_name(name: str) -> str:
    """
    Validate a model name contains only safe characters.
    
    Args:
        name: Model name to validate
        
    Returns:
        Validated model name
        
    Raises:
        PathSecurityError: If name contains unsafe characters
    """
    # Allow alphanumeric, hyphens, underscores, and dots
    if not re.match(r'^[\w\-\.]+$', name):
        raise PathSecurityError(
            f"Model name contains unsafe characters: {name}. "
            "Only alphanumeric, hyphen, underscore, and dot allowed."
        )
    return name


def safe_join(base: Union[str, Path], *paths: Union[str, Path]) -> Path:
    """
    Safely join paths, preventing directory traversal.
    
    Args:
        base: Base directory
        *paths: Path components to join
        
    Returns:
        Joined and validated path
        
    Raises:
        PathSecurityError: If resulting path escapes base directory
    """
    base = Path(base).expanduser().resolve()
    result = base.joinpath(*paths).resolve()
    
    # Ensure result is still within base
    try:
        result.relative_to(base)
    except ValueError:
        raise PathSecurityError(
            f"Joined path {result} escapes base directory {base}"
        )
    
    return result
