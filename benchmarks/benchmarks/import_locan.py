"""
Benchmark functions to be used with Airspeed Velocity.
"""


def timeraw_import_locan():
    """Benchmark import time for locan"""
    return """
    import locan
    """
