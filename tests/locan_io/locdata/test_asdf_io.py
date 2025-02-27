import tempfile
from pathlib import Path

from pandas.testing import assert_frame_equal

from locan import PROPERTY_KEYS, FileType
from locan.locan_io import convert_property_types, load_asdf_file, save_asdf


def test_save_and_load_asdf(locdata_2d):
    # for visual inspection use:
    # io.save_asdf(locdata_2d, path=locan.ROOT_DIR / 'tests/test_data/locdata.asdf')
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "locdata.asdf"
        save_asdf(locdata_2d, path=file_path)

        locdata = load_asdf_file(path=file_path, convert=False)
        locdata_data = convert_property_types(locdata.data, types=PROPERTY_KEYS)
        assert_frame_equal(locdata_data, locdata_2d.data)
        assert locdata.meta.identifier == locdata_2d.meta.identifier
        assert locdata.properties == locdata_2d.properties
        assert locdata.meta.file.type == FileType.ASDF.value
        assert locdata.meta.file.path == str(file_path)

        dat = load_asdf_file(path=file_path, nrows=5)
        assert len(dat) == 5
