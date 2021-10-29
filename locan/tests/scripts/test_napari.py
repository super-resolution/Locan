import os
from pathlib import Path

import pytest
from locan.constants import _has_napari
if _has_napari: import napari

from locan.constants import ROOT_DIR
from locan.scripts.script_napari import sc_napari


@pytest.mark.gui
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_napari():
    viewer = napari.Viewer(show=False)
    assert viewer


@pytest.mark.gui
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_script_napari():
    path = Path(ROOT_DIR / 'tests/test_data/five_blobs.txt')
    assert path.exists()
    sc_napari(file_path=str(path), file_type=1, bin_size=20, rescale=None)


@pytest.mark.gui
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_script_napari_from_sys():
    path = Path(ROOT_DIR / 'tests/test_data/five_blobs.txt')
    assert path.exists()
    exit_status = os.system(f'locan napari -f {str(path)} -t 1 --bin_size 20 --rescale None')
    assert exit_status == 0
