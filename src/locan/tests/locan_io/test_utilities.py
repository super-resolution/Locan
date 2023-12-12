from locan import ROOT_DIR, find_file_upstream


def test_find_file_upstream():
    # find file
    sub_directory = ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    pattern = None
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found.parent == sub_directory.parent

    # find file
    sub_directory = ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    pattern = "*.txt"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found.parent == sub_directory.parent

    sub_directory = ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found == ROOT_DIR / "tests/__init__.py"

    sub_directory = ROOT_DIR / "tests/test_data/"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found == ROOT_DIR / "tests/__init__.py"

    sub_directory = ROOT_DIR / "tests/test_data/"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(
        sub_directory=sub_directory, pattern=pattern, regex="init"
    )
    assert pattern_found == ROOT_DIR / "tests/__init__.py"

    sub_directory = ROOT_DIR / "tests/test_data/"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(
        sub_directory=sub_directory, pattern=pattern, regex="not_existing"
    )
    assert pattern_found is None

    # find directory
    sub_directory = ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    pattern = "tests"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found == ROOT_DIR / "tests"

    # find no file
    sub_directory = ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    pattern = "not_present.any"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found is None

    sub_directory = ROOT_DIR / "tests/test_data/Elyra_dstorm_data.txt"
    pattern = "__init__.py"
    directory = ROOT_DIR / "tests/test_data"
    pattern_found = find_file_upstream(
        sub_directory=sub_directory, pattern=pattern, top_directory=directory
    )
    assert pattern_found is None
