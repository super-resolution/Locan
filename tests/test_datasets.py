import pytest

from locan.datasets import load_npc, load_tubulin

httpx = pytest.importorskip("httpx", reason="requires httpx for html download")


def test_load_npc():
    locdata = load_npc()
    assert locdata.meta.element_count == 2285189


def test_load_tubulin():
    locdata = load_tubulin()
    assert locdata.meta.element_count == 1506568
