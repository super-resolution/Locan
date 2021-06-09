import os
from pathlib import Path

import pytest
from locan.constants import _has_napari
if _has_napari: import napari

from locan.constants import ROOT_DIR
from locan.scripts.script_rois import sc_draw_roi_napari


@pytest.mark.skip('GUI tests are skipped because they need user interaction.')
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_script_rois(capfd):
    path = Path(ROOT_DIR / 'tests/test_data/five_blobs.txt')
    assert path.exists()
    roi_path_list = sc_draw_roi_napari(file_path=str(path), file_type=1, bin_size=20, rescale=None)
    captured = capfd.readouterr()
    print(captured.out)
    assert '[Roi' in captured.out

    # delete roi files
    for path in roi_path_list:
        Path(path).unlink()