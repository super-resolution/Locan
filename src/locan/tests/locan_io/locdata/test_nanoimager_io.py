from io import StringIO

import locan.constants
from locan import FileType
from locan.locan_io import load_Nanoimager_file
from locan.locan_io.locdata.nanoimager_io import load_Nanoimager_header


def test_get_correct_column_names_from_Nanoimager_header():
    columns = load_Nanoimager_header(
        path=locan.ROOT_DIR / "tests/test_data/Nanoimager_dstorm_data.csv"
    )
    assert columns == [
        "channel",
        "frame",
        "position_x",
        "position_y",
        "position_z",
        "intensity",
        "local_background",
    ]

    file_like = StringIO(
        "Channel,Frame,X (nm),Y (nm),Z (nm),Photons,Background\n"
        "0,1548,40918.949219,56104.691406,0.000000,139.828232,0.848500"
    )
    columns = load_Nanoimager_header(path=file_like)
    assert columns == [
        "channel",
        "frame",
        "position_x",
        "position_y",
        "position_z",
        "intensity",
        "local_background",
    ]


def test_loading_Nanoimager_file():
    file_path = locan.ROOT_DIR / "tests/test_data/Nanoimager_dstorm_data.csv"
    dat = load_Nanoimager_file(path=file_path, nrows=10)
    # print(dat.data.columns)
    assert len(dat) == 10
    assert all(
        dat.data.columns
        == [
            "channel",
            "frame",
            "position_x",
            "position_y",
            "position_z",
            "intensity",
            "local_background",
        ]
    )
    assert dat.meta.file.type == FileType.NANOIMAGER.value
    assert dat.meta.file.path == str(file_path)

    file_like = StringIO(
        "Channel,Frame,X (nm),Y (nm),Z (nm),Photons,Background\n"
        "0,1548,40918.949219,56104.691406,0.000000,139.828232,0.848500"
    )
    dat = load_Nanoimager_file(path=file_like, nrows=1)
    assert len(dat) == 1
    assert all(
        dat.data.columns
        == [
            "channel",
            "frame",
            "position_x",
            "position_y",
            "position_z",
            "intensity",
            "local_background",
        ]
    )
