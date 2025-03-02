from locan import find_file_upstream
from tests import TEST_DIR


def test_find_file_upstream():
    # find file
    sub_directory = TEST_DIR / "test_data/Elyra_dstorm_data.txt"
    pattern = None
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found.parent.resolve() == sub_directory.parent.resolve()

    # find file
    sub_directory = TEST_DIR / "test_data/Elyra_dstorm_data.txt"
    pattern = "*.txt"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found.parent.resolve() == sub_directory.parent.resolve()

    sub_directory = TEST_DIR / "test_data/Elyra_dstorm_data.txt"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found == TEST_DIR.resolve() / "__init__.py"

    sub_directory = TEST_DIR / "test_data/"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found == TEST_DIR.resolve() / "__init__.py"

    sub_directory = TEST_DIR / "test_data/"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(
        sub_directory=sub_directory, pattern=pattern, regex="init"
    )
    assert pattern_found == TEST_DIR.resolve() / "__init__.py"

    sub_directory = TEST_DIR / "test_data/"
    pattern = "__init__.py"
    pattern_found = find_file_upstream(
        sub_directory=sub_directory, pattern=pattern, regex="not_existing"
    )
    assert pattern_found is None

    # find directory
    sub_directory = TEST_DIR / "test_data/Elyra_dstorm_data.txt"
    pattern = "tests"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found == TEST_DIR.resolve()

    # find no file
    sub_directory = TEST_DIR / "test_data/Elyra_dstorm_data.txt"
    pattern = "not_present.any"
    pattern_found = find_file_upstream(sub_directory=sub_directory, pattern=pattern)
    assert pattern_found is None

    sub_directory = TEST_DIR / "test_data/Elyra_dstorm_data.txt"
    pattern = "__init__.py"
    directory = TEST_DIR.resolve() / "test_data"
    pattern_found = find_file_upstream(
        sub_directory=sub_directory, pattern=pattern, top_directory=directory
    )
    assert pattern_found is None
