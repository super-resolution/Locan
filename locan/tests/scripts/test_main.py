import os


def test_main(capfd):
    exit_status = os.system(f'locan --help')
    captured = capfd.readouterr()
    assert captured.out.startswith("usage:")
    assert exit_status == 0

    exit_status = os.system(f'locan --version')
    captured = capfd.readouterr()
    assert captured.out.startswith("locan version")
    assert exit_status == 0
