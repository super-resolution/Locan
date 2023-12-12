from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from locan import Files


@pytest.fixture(scope="session")
def test_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("test_directory")
    (directory / "sub_directory").mkdir()
    files = [
        directory / "sub_directory" / "file_group_a_0.data",
        directory / "sub_directory" / "file_group_a_1.data",
        directory / "sub_directory" / "file_group_b_2.data",
        directory / "sub_directory" / "corresponding_file_0.data",
        directory / "metadata.meta",
    ]
    for file_ in files:
        file_.touch()
    return directory


def test_fixture_test_files(test_files):
    files = list(test_files.glob("**/*.*"))
    assert len(files) == 5


def test_Files_init(test_files, capfd):
    files = Files()
    assert files.directory is None
    assert "file_path" in files.df.columns
    assert isinstance(files.df, pd.DataFrame)
    assert len(files.df) == 0
    assert files.print_summary() is None

    with pytest.raises(ValueError):
        Files(directory=test_files / "does_not_exist")

    files = Files(
        directory=test_files,
        df={"my_file": [test_files / "sub_directory" / "file_group_a_0.data"]},
        column="my_file",
    )
    assert files.directory == test_files
    assert isinstance(files.df, pd.DataFrame)
    assert "my_file" in files.df.columns
    assert isinstance(files.df.my_file[0], Path)
    assert len(files.df) == 1
    assert files.print_summary() is None

    with pytest.raises(FileExistsError):
        Files(df={"file_path": [test_files / "does_not_exist.data"]})

    files = Files(df={"file_path": [test_files / "does_not_exist.data"]}, exists=False)
    assert files.directory is None
    assert "file_path" in files.df.columns
    assert isinstance(files.df, pd.DataFrame)
    assert isinstance(files.df.file_path[0], Path)
    assert len(files.df) == 1
    assert files.print_summary() is None

    capfd.readouterr()


def test_Files_from_path(test_files, capfd):
    file_path = test_files / "sub_directory" / "file_group_a_0.data"
    files = Files.from_path(file_path)
    assert files.directory is None
    assert isinstance(files.df, pd.DataFrame)
    assert isinstance(files.df.file_path[0], Path)
    assert len(files.df) == 1
    assert files.print_summary() is None

    files = Files.from_path(str(file_path))
    assert files.directory is None
    assert isinstance(files.df, pd.DataFrame)
    assert isinstance(files.df.file_path[0], Path)
    assert len(files.df) == 1
    assert files.print_summary() is None

    file_path = [
        test_files / "sub_directory" / "file_group_a_0.data",
        test_files / "sub_directory" / "file_group_a_1.data",
    ]

    files = Files.from_path(file_path)
    assert files.directory is None
    assert isinstance(files.df, pd.DataFrame)
    assert isinstance(files.df.file_path[0], Path)
    assert len(files.df) == 2
    assert files.print_summary() is None

    files = Files.from_path([str(f_) for f_ in file_path])
    assert files.directory is None
    assert isinstance(files.df, pd.DataFrame)
    assert isinstance(files.df.file_path[0], Path)
    assert len(files.df) == 2
    assert files.print_summary() is None

    capfd.readouterr()


def test_Files_indexing(test_files, capfd):
    file_path = [
        test_files / "sub_directory" / "file_group_a_0.data",
        test_files / "sub_directory" / "file_group_a_1.data",
    ]

    files = Files.from_path(file_path)
    # test __iter__:
    for file_ in files:
        assert isinstance(file_, tuple)
        assert file_.file_path

    # test __getitem__:
    file_ = files[0]
    assert isinstance(file_, pd.Series)
    assert file_.file_path

    file_ = files[0:2]
    assert isinstance(file_, Files)
    assert len(file_.df.file_path) == 2

    file_ = files[0:1]
    assert isinstance(file_, Files)
    assert len(file_.df.file_path) == 1

    file_ = files[[0]]
    assert isinstance(file_, Files)
    assert len(file_.df.file_path) == 1


def test_Files_from_glob(test_files, capfd):
    files = Files().from_glob(pattern="this_should_not_ever_exist_#")
    assert files.directory.resolve() == Path().cwd().resolve()
    assert files.df.empty

    files = Files().from_glob(directory=test_files)
    assert files.directory.resolve() == test_files.resolve()
    assert files.df.empty

    files = Files().from_glob(directory=test_files, pattern="**/*.data")
    assert files.directory.resolve() == test_files.resolve()
    assert len(files.df) == 4

    files = Files().from_glob(
        directory=test_files, pattern="**/*.data", regex="corresponding"
    )
    assert files.directory.resolve() == test_files.resolve()
    assert len(files.df) == 1


def test_Files_add_glob(test_files, capfd, caplog):
    files = Files().from_glob(directory=test_files, pattern="**/*file_group_a_0.*")
    files.add_glob(pattern="**/*_0.*", regex="corresponding")
    assert len(files.df) == 1
    assert "file_path" in files.df.columns
    assert "other_file_path" in files.df.columns

    files = Files().from_glob(directory=test_files, pattern="**/*file_group_a*.*")
    files.add_glob(pattern="**/*_0.*", regex="corresponding")
    assert len(files.df) == 2
    assert "file_path" in files.df.columns
    assert "other_file_path" in files.df.columns
    assert caplog.record_tuples == [
        ("locan.locan_io.files", 30, "Not all files are matched.")
    ]


def test_Files_match_files(test_files, capfd, caplog):
    files = Files().from_glob(directory=test_files, pattern="**/*file_group_a_0.*")
    files_match = Files().from_glob(
        directory=test_files, pattern="**/*_0.*", regex="corresponding"
    )
    files.match_files(files=files_match.df["file_path"])
    assert len(files.df) == 1
    assert "file_path" in files.df.columns
    assert "other_file_path" in files.df.columns

    files = Files().from_glob(directory=test_files, pattern="**/*file_group_a*.*")
    files_match = Files().from_glob(
        directory=test_files, pattern="**/*_0.*", regex="corresponding"
    )
    files.match_files(files=files_match.df["file_path"])
    assert len(files.df) == 2
    assert "file_path" in files.df.columns
    assert "other_file_path" in files.df.columns
    assert caplog.record_tuples == [
        ("locan.locan_io.files", 30, "Not all files are matched.")
    ]


def test_Files_exclude(test_files, capfd):
    files = Files().from_glob(directory=test_files, pattern="**/*.*")
    assert len(files.df) == 5

    files_new = deepcopy(files)
    stoplist = None
    files_new = files_new.exclude(stoplist=stoplist)
    assert len(files_new.df) == len(files.df)
    assert "file_path" in files_new.df.columns

    files_new = deepcopy(files)
    stoplist = []
    files_new = files_new.exclude(stoplist=stoplist)
    assert len(files_new.df) == len(files.df)
    assert "file_path" in files_new.df.columns

    files_new = deepcopy(files)
    stoplist = Files()
    files_new = files_new.exclude(stoplist=stoplist)
    assert len(files_new.df) == len(files.df)
    assert "file_path" in files_new.df.columns

    files_new = deepcopy(files)
    stoplist = [False, True, True, True, True]
    files_new = files_new.exclude(stoplist=stoplist)
    assert len(files_new.df) == 1
    assert "file_path" in files_new.df.columns

    with pytest.raises(ValueError):
        stoplist = [False, True]
        files_new.exclude(stoplist=stoplist)

    files_new = deepcopy(files)
    stoplist = Files().from_glob(directory=test_files, pattern="**/*.data")
    stoplist = list(stoplist.df.file_path)
    files_new = files_new.exclude(stoplist=stoplist)
    assert len(files_new.df) == 1

    files_new = deepcopy(files)
    stoplist = Files().from_glob(directory=test_files, pattern="**/*.data")
    stoplist = list(stoplist.df.file_path.astype("string"))
    files_new = files_new.exclude(stoplist=stoplist)
    assert len(files_new.df) == 1

    files_new = deepcopy(files)
    stoplist = Files().from_glob(directory=test_files, pattern="**/*.data")
    files_new = files_new.exclude(stoplist=stoplist)
    assert len(files_new.df) == 1

    files_new = deepcopy(files)
    stoplist = Files().from_glob(
        directory=test_files, pattern="**/*.data", column="stoplist"
    )
    files_new = files_new.exclude(stoplist=stoplist, column_stoplist="stoplist")
    assert len(files_new.df) == 1

    files_new = deepcopy(files)
    files_new = files_new.exclude(stoplist=files)
    assert len(files_new.df) == 0


def test_Files_concatenate(test_files, capfd):
    files_1 = Files().from_glob(directory=test_files, pattern="**/*.data", column="one")
    files_2 = Files().from_glob(directory=test_files, pattern="**/*.meta", column="one")
    files = Files().concatenate(files=[files_1, files_2], exists=False)
    assert len(files.df) == 5
    assert "one" in files.df.columns


def test_Files_match_file_upstream(test_files, capfd):
    files = Files().from_glob(directory=test_files, pattern="**/*.data")
    files_new = files.match_file_upstream(
        pattern="*.meta", other_column="matching_file"
    )
    assert len(files_new.df) == 4
    assert len(files_new.df.columns) == 2

    files = Files().from_glob(directory=test_files, pattern="**/*.data")
    files_new = files.match_file_upstream(
        pattern="*.data", other_column="matching_file", regex="corresponding"
    )
    assert len(files_new.df) == 4
    assert len(files_new.df.columns) == 2
    assert files_new.df.matching_file.notna().any()


def test_set_group_identifier(test_files, capfd):
    files = Files().from_glob(directory=test_files, pattern="**/*.*")
    # print(files.files.applymap(lambda x: x.name))
    files.set_group_identifier(name="group_subdir", pattern="sub_directory")
    assert files.df.group[files.df.group == "group_subdir"].count() == 4
    assert files.df.group.dtype == "category"
    files.set_group_identifier(name="group_data", glob="*.data")
    assert files.df.group[files.df.group == "group_data"].count() == 4
    assert files.df.group.dtype == "category"
    files.set_group_identifier(name="group_b", regex="group_b")
    assert files.df.group[files.df.group == "group_b"].count() == 1
    assert files.df.group.dtype == "category"
    files = Files().from_glob(directory=test_files, pattern="**/*.*")
    files.set_group_identifier(
        name="group_b", pattern="sub_directory", glob="*.txt", regex="group_b"
    )
    assert files.df.group[files.df.group == "group_b"].count() == 0
    assert files.df.group.dtype == "category"
    files.set_group_identifier(
        name="group_b", pattern="sub_directory", glob="*.data", regex="group_b"
    )
    assert files.df.group[files.df.group == "group_b"].count() == 1
    assert files.df.group.dtype == "category"

    assert "group_b" in files.group_identifiers()
    assert isinstance(files.grouped(), pd.core.groupby.DataFrameGroupBy)
