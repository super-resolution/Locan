"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from surepy.scripts import show_versions

pytestmark = pytest.mark.skip('GUI tests are skipped because they would need user interaction.')


def test_script_show_version():
    show_versions.main()
