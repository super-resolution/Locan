from io import StringIO

import locan.constants
from locan.locan_io import load_thunderstorm_file
from locan.locan_io.locdata.thunderstorm_io import load_thunderstorm_header


def test_get_correct_column_names_from_Thunderstorm_header():
    columns = load_thunderstorm_header(
        path=locan.ROOT_DIR / "tests/test_data/Thunderstorm_dstorm_data.csv"
    )
    assert columns == [
        "original_index",
        "frame",
        "position_x",
        "position_y",
        "psf_sigma",
        "intensity",
        "local_background",
        "local_background_sigma",
        "chi_square",
        "uncertainty",
    ]

    file_like = StringIO(
        "id,frame,x [nm],y [nm]\n" "73897.0,2001.0,1320.109670647555,26344.7124618434"
    )
    columns = load_thunderstorm_header(path=file_like)
    assert columns == ["original_index", "frame", "position_x", "position_y"]


def test_loading_Thunderstorm_file():
    dat = load_thunderstorm_file(
        path=locan.ROOT_DIR / "tests/test_data/Thunderstorm_dstorm_data.csv", nrows=10
    )
    # print(dat.data.columns)
    assert len(dat) == 10

    file_like = StringIO(
        "id,frame,x [nm],y [nm]\n" "73897.0,2001.0,1320.109670647555,26344.7124618434"
    )
    dat = load_thunderstorm_file(path=file_like, nrows=1)
    assert len(dat) == 1
