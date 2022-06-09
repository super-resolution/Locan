from locan import dependency_info, show_versions, system_info


def test__get_sys_info():
    sys_info = system_info()
    # print(sys_info)
    assert len(sys_info) == 12


def test___get_dependency_info():
    deps_info = dependency_info(other_dependencies=["locan"])
    # print(deps_info)
    assert "boost-histogram" in deps_info
    assert "locan" in deps_info


def test_show_versions_(capfd):
    show_versions(other_dependencies=["locan"])
    captured = capfd.readouterr()
    for header in [
        "\nLocan:",
        "\nPython:",
        "\nSystem:",
        "\nPython dependencies:",
        "locan",
        "node:",
        "executable:",
    ]:
        assert header in captured.out

    show_versions(verbose=False, extra_dependencies=False)
    captured = capfd.readouterr()
    for header in ["\nLocan:", "\nPython:", "\nSystem:", "\nPython dependencies:"]:
        assert header in captured.out
    for header in ["node:", "executable:"]:
        assert header not in captured.out
