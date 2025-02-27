from locan import ROOT_DIR, locdata_id


def test_root_directory():
    assert ROOT_DIR.is_dir()
    # print(ROOT_DIR.joinpath('tests/'))
    # print(type(ROOT_DIR))
    assert ROOT_DIR.joinpath("tests/").is_dir()


def test_locdata_id():
    assert locdata_id == 0
