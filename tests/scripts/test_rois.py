from pathlib import Path

import pytest

from locan import ROOT_DIR
from locan.dependencies import HAS_DEPENDENCY
from locan.scripts.script_rois import sc_draw_roi_napari


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_script_rois(capfd):
    path = Path(ROOT_DIR / "tests/test_data/five_blobs.txt")
    assert path.exists()
    roi_path_list = sc_draw_roi_napari(
        file_path=str(path), file_type=1, bin_size=20, rescale=None
    )
    captured = capfd.readouterr()
    print(captured.out)
    assert "[Roi" in captured.out

    # delete roi files
    for path in roi_path_list:
        Path(path).unlink()
