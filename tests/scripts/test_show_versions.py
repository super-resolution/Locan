from locan.scripts import script_show_versions


def test_script_show_version(capfd):
    script_show_versions.main([])
    captured = capfd.readouterr()
    for header in ["\nLocan:", "\nPython:", "\nSystem:", "\nPython dependencies:"]:
        assert header in captured.out
    for header in ["node:", "executable:", "locan"]:
        assert header not in captured.out

    script_show_versions.main(["-o", "locan"])
    captured = capfd.readouterr()
    for header in [
        "\nLocan:",
        "\nPython:",
        "\nSystem:",
        "\nPython dependencies:",
        "locan",
    ]:
        assert header in captured.out

    script_show_versions.main(["-v", "-e", "-o", "locan"])
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
