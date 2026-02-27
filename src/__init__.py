"""KV-Cloak shared utilities package."""

from .config import (
    MODEL_CONFIGS,
    KVCLOAK_DEFAULTS,
    DP_DEFAULTS,
    ATTACK_DEFAULTS,
    BITTER_LESSON_TEXT,
    get_model_path,
    validate_model_name,
    get_cache_path,
)

from .security_utils import (
    validate_path,
    validate_model_name,
    safe_join,
    PathSecurityError,
)

__all__ = [
    # Config exports
    "MODEL_CONFIGS",
    "KVCLOAK_DEFAULTS",
    "DP_DEFAULTS",
    "ATTACK_DEFAULTS",
    "BITTER_LESSON_TEXT",
    "get_model_path",
    "validate_model_name",
    "get_cache_path",
    # Security exports
    "validate_path",
    "safe_join",
    "PathSecurityError",
]
