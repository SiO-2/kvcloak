"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_dir(project_root):
    """Return the src directory."""
    return project_root / "src"


@pytest.fixture(scope="session")
def tests_dir(project_root):
    """Return the tests directory."""
    return project_root / "tests"
