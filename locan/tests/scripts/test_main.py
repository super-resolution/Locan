import os


def test_main(capfd):
    exit_status = os.system(f"locan --help")
    captured = capfd.readouterr()
    assert captured.out.startswith("usage:")
    assert exit_status == 0

    exit_status = os.system(f"locan --version")
    captured = capfd.readouterr()
    assert captured.out.startswith("locan version")
    assert exit_status == 0

    exit_status = os.system(f"locan --info")
    captured = capfd.readouterr()
    for item in ["Locan:", "Python:", "System:", "Python dependencies:"]:
        assert item in captured.out
    assert exit_status == 0
