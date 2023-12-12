from io import StringIO

import locan.constants
from locan import FileType
from locan.locan_io import load_Elyra_file
from locan.locan_io.locdata.elyra_io import load_Elyra_header


def test_get_correct_column_names_from_Elyra_header():
    columns = load_Elyra_header(
        path=locan.ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    )
    assert columns == [
        "original_index",
        "frame",
        "frames_number",
        "frames_missing",
        "position_x",
        "position_y",
        "uncertainty",
        "intensity",
        "local_background_sigma",
        "chi_square",
        "psf_half_width",
        "channel",
        "slice_z",
    ]

    file_like = StringIO(
        "Index	First Frame	Number Frames	Frames Missing	Position X [nm]	Position Y [nm]\n"
        "1  1   1   0   15850.6 23502.1"
    )
    columns = load_Elyra_header(path=file_like)
    assert columns == [
        "original_index",
        "frame",
        "frames_number",
        "frames_missing",
        "position_x",
        "position_y",
    ]


def test_loading_Elyra_file():
    file_path = locan.ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    dat = load_Elyra_file(path=file_path)
    # loading is not limited by nrows=10 to ensure correct treatment of file appendix and NUL character.
    assert len(dat) == 999
    assert dat.meta.file.type == FileType.ELYRA.value
    assert dat.meta.file.path == str(file_path)

    file_like = StringIO(
        "Index\tFirst Frame\tNumber Frames\tFrames Missing\tPosition X [nm]\tPosition Y [nm]\n"
        "1\t1\t1\t0\t15850.6\t23502.1"
    )
    dat = load_Elyra_file(path=file_like)
    # loading is not limited by nrows=10 to ensure correct treatment of file appendix and NUL character.
    assert len(dat) == 1
