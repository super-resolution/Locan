import os
from pathlib import Path
import tifffile as tif

import pytest
from locan.constants import _has_napari
if _has_napari: import napari

from locan.constants import ROOT_DIR
from locan import load_locdata
from locan.scripts.script_check import render_locs_per_frame_napari, sc_check


@pytest.mark.gui
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_render_locs_per_frame_napari(locdata_2d):
    images_path = Path(ROOT_DIR / 'tests/test_data/images.tif')
    assert images_path.exists()
    locdata_path = Path(ROOT_DIR / 'tests/test_data/rapidSTORM_from_images.txt')
    assert locdata_path.exists()

    image_stack = tif.imread(str(images_path))
    locdata = load_locdata(locdata_path, file_type=2)

    viewer = render_locs_per_frame_napari(images=image_stack, pixel_size=133, locdata=locdata,
                                          viewer=None, transpose=True,
                                          kwargs_image=None, kwargs_points=None)
    assert viewer
    # napari.run()


@pytest.mark.gui
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_script_check():
    images_path = Path(ROOT_DIR / 'tests/test_data/images.tif')
    assert images_path.exists()
    locdata_path = Path(ROOT_DIR / 'tests/test_data/rapidSTORM_from_images.txt')
    assert locdata_path.exists()
    sc_check(pixel_size=133, file_images=images_path, file_locdata=locdata_path, file_type=2)


@pytest.mark.gui
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_script_check_from_sys(capfd):
    images_path = Path(ROOT_DIR / 'tests/test_data/images.tif')
    assert images_path.exists()
    locdata_path = Path(ROOT_DIR / 'tests/test_data/rapidSTORM_from_images.txt')
    assert locdata_path.exists()
    exit_status = os.system(f'locan check 133 -f {str(images_path)} -l {str(locdata_path)} -t 2')
    captured = capfd.readouterr()
    print(captured.out)
    assert exit_status == 0
