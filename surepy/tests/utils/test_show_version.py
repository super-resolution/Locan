import pytest

from surepy.utils.system_information import system_info, dependency_info, show_versions


def test__get_sys_info():
    sys_info = system_info()
    # print(sys_info)
    assert len(sys_info) == 13


def test___get_dependency_info():
    deps_info = dependency_info()
    # print(deps_info)
    assert 'numpy' in deps_info


@pytest.mark.skip('Test requires visual inspection of printout.')
def test_show_versions():
    show_versions()