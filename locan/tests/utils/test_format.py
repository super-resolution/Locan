import time

from locan.utils.format import _time_string


def test_time_string():
    time_ = 1000000000  # result from time.time()
    string_ = _time_string(time_)
    assert string_ == '2001-09-09 03:46:40 +0200'
