import pytest

from surepy.constants import DATASETS_DIR
from surepy.datasets import load_npc

pytestmark = pytest.mark.skip('These tests are skipped because they require data in Surepy_datasets directory.')

def test_load_npc():
    assert DATASETS_DIR.exists()
    locdata = load_npc()
    assert locdata.meta.element_count == 2285189
