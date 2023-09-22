from io import StringIO

import numpy as np

import locan.constants
from locan.locan_io import load_rapidSTORM_file, load_rapidSTORM_track_file
from locan.locan_io.locdata.rapidstorm_io import (
    load_rapidSTORM_header,
    load_rapidSTORM_track_header,
)


def test_get_correct_column_names_from_rapidSTORM_header():
    columns = load_rapidSTORM_header(
        path=locan.ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_data.txt"
    )
    assert columns == [
        "position_x",
        "position_y",
        "frame",
        "intensity",
        "chi_square",
        "local_background",
    ]

    file_like = StringIO(
        '# <localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><field identifier="FitResidues-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="fit residue chi square value" unit="dimensionless" /><field identifier="LocalBackground-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="local background" unit="A/D count" /></localizations>\n'
        "9657.4 24533.5 0 33290.1 1.19225e+006 767.733"
    )
    columns = load_rapidSTORM_header(path=file_like)
    assert columns == [
        "position_x",
        "position_y",
        "frame",
        "intensity",
        "chi_square",
        "local_background",
    ]


def test_loading_rapidSTORM_file():
    path = locan.ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_data.txt"
    dat = load_rapidSTORM_file(path=path, nrows=10)
    # print(dat.data.head())
    # dat.print_meta()
    assert len(dat) == 10
    assert dat.meta.file.type == locan.FileType.RAPIDSTORM.value
    assert dat.meta.file.path == str(path)

    file_like = StringIO(
        '# <localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="3.27165e-005 m" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><field identifier="FitResidues-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="fit residue chi square value" unit="dimensionless" /><field identifier="LocalBackground-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="local background" unit="A/D count" /></localizations>\n'
        "9657.4 24533.5 0 33290.1 1.19225e+006 767.733"
    )
    dat = load_rapidSTORM_file(path=file_like, nrows=1)
    assert len(dat) == 1


def test_get_correct_column_names_from_rapidSTORM_track_header():
    columns = load_rapidSTORM_track_header(
        path=locan.ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_track_data.txt"
    )
    assert columns == (
        ["position_x", "position_y", "frame", "intensity"],
        [
            "position_x",
            "uncertainty_x",
            "position_y",
            "uncertainty_y",
            "frame",
            "intensity",
            "chi_square",
            "local_background",
        ],
    )

    file_like = StringIO(
        '# <localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="Position-0-0-uncertainty" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position uncertainty in sample space in X" unit="nanometer" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="Position-1-0-uncertainty" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position uncertainty in sample space in Y" unit="nanometer" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><field identifier="FitResidues-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="fit residue chi square value" unit="dimensionless" /><field identifier="LocalBackground-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="local background" unit="A/D count" /></localizations></localizations>\n'
        "5417.67 8439.85 1 4339.29 2 5421.22 25 8440.4 25 0 2292.39 48696.1 149.967 5414.13 25 8439.3 25 1 2046.9 65491.4 142.521"
    )
    columns = load_rapidSTORM_track_header(path=file_like)
    assert columns == (
        ["position_x", "position_y", "frame", "intensity"],
        [
            "position_x",
            "uncertainty_x",
            "position_y",
            "uncertainty_y",
            "frame",
            "intensity",
            "chi_square",
            "local_background",
        ],
    )


def test_loading_rapidSTORM_track_file():
    dat = load_rapidSTORM_track_file(
        path=locan.ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_track_data.txt",
        min_localization_count=2,
        nrows=10,
    )
    # print(dat.data.head())
    # print(dat.data.columns)
    # dat.print_meta()
    assert np.array_equal(
        dat.data.columns,
        [
            "localization_count",
            "position_x",
            "uncertainty_x",
            "position_y",
            "uncertainty_y",
            "intensity",
            "local_background",
            "frame",
            "region_measure_bb",
            "localization_density_bb",
            "subregion_measure_bb",
        ],
    )
    assert len(dat) == 9
    # len(dat) is 9 and not 10 since one row is filtered out y min_localization_count=2

    dat = load_rapidSTORM_track_file(
        path=locan.ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_track_data.txt",
        collection=False,
        nrows=10,
    )
    assert np.array_equal(
        dat.data.columns, ["position_x", "position_y", "frame", "intensity"]
    )
    assert len(dat) == 10

    file_like = StringIO(
        '# <localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><localizations insequence="true" repetitions="variable"><field identifier="Position-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in X" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="Position-0-0-uncertainty" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position uncertainty in sample space in X" unit="nanometer" /><field identifier="Position-1-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position in sample space in Y" unit="nanometer" min="0 m" max="8.442e-006 m" /><field identifier="Position-1-0-uncertainty" syntax="floating point with . for decimals and optional scientific e-notation" semantic="position uncertainty in sample space in Y" unit="nanometer" /><field identifier="ImageNumber-0-0" syntax="integer" semantic="frame number" unit="frame" min="0 fr" /><field identifier="Amplitude-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="emission strength" unit="A/D count" /><field identifier="FitResidues-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="fit residue chi square value" unit="dimensionless" /><field identifier="LocalBackground-0-0" syntax="floating point with . for decimals and optional scientific e-notation" semantic="local background" unit="A/D count" /></localizations></localizations>\n'
        "5417.67 8439.85 1 4339.29 2 5421.22 25 8440.4 25 0 2292.39 48696.1 149.967 5414.13 25 8439.3 25 1 2046.9 65491.4 142.521"
    )
    dat = load_rapidSTORM_file(path=file_like, nrows=1)
    assert len(dat) == 1
