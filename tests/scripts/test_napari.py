import subprocess

import pytest

from locan.dependencies import HAS_DEPENDENCY
from locan.scripts.script_napari import sc_napari
from tests import TEST_DIR


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_script_napari():
    path = TEST_DIR / "test_data/five_blobs.txt"
    assert path.exists()
    sc_napari(file_path=str(path), file_type=1, bin_size=20, rescale=None)


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_script_napari_from_sys():
    path = TEST_DIR / "test_data/five_blobs.txt"
    assert path.exists()
    exit_status = subprocess.run(  # noqa S603
        f"locan napari -f {str(path)} -t 1 --bin_size 20 --rescale None",
    )
    exit_status.check_returncode()


# pass to CL:
# locan napari -f "tests/test_data/five_blobs.txt" -t 1 --bin_size 20 --rescale None
