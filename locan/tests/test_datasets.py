import pytest

from locan.datasets import load_npc, load_tubulin
from locan.dependencies import HAS_DEPENDENCY

# pytestmark = pytest.mark.requires_datasets
pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCY["requests"], reason="requires requests for html download"
)


def test_load_npc():
    locdata = load_npc()
    assert locdata.meta.element_count == 2285189


def test_load_tubulin():
    locdata = load_tubulin()
    assert locdata.meta.element_count == 1506568
