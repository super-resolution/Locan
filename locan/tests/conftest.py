import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import locan.data.metadata_pb2
from locan import ROOT_DIR, LocData
from locan.dependencies import HAS_DEPENDENCY
from locan.locan_io.locdata.io_locdata import load_rapidSTORM_file, load_txt_file

logger = logging.getLogger(__name__)

for key, value in HAS_DEPENDENCY.items():
    if not value:
        logger.info(f"Extra dependency {key} is not available.")


# register pytest markers - should be in sync with pyproject.toml
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gui: marks tests that require user interaction (skipped by default)"
    )
    config.addinivalue_line(
        "markers",
        "visual: marks tests that require visual inspection (skipped by default)",
    )
    config.addinivalue_line(
        "markers",
        "requires_datasets: marks tests that require data in locan_datasets directory (skipped by default)",
    )


# fixtures for random points


@pytest.fixture(scope="session")
def few_random_points():
    points = np.array(
        [
            [0.066, 0.64],
            [0.92, 0.65],
            [0.11, 0.40],
            [0.20, 0.17],
            [0.75, 0.92],
            [0.01, 0.12],
            [0.23, 0.54],
            [0.05, 0.25],
            [0.70, 0.73],
            [0.43, 0.16],
        ]
    )
    return points


# fixtures for LocData objects


@pytest.fixture(scope="session")
def locdata_empty():
    df = pd.DataFrame()
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_single_localization():
    locdata_dict = {
        "position_x": np.array([1]),
        "position_y": np.array([1]),
        "frame": np.array([1]),
        "intensity": np.array([1]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_single_localization_3d():
    locdata_dict = {
        "position_x": np.array([1]),
        "position_y": np.array([1]),
        "position_z": np.array([1]),
        "frame": np.array([1]),
        "intensity": np.array([1]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_1d():
    locdata_dict = {
        "position_x": np.array([1, 1, 2, 3, 4, 5]),
        "frame": np.array([1, 2, 2, 4, 5, 6]),
        "intensity": np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_2d():
    locdata_dict = {
        "position_x": np.array([1, 1, 2, 3, 4, 5]),
        "position_y": np.array([1, 5, 3, 6, 2, 5]),
        "frame": np.array([1, 2, 2, 4, 5, 6]),
        "intensity": np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_2d_negative():
    locdata_dict = {
        "position_x": np.array([1, -1, 2, 3, 4, 5]),
        "position_y": np.array([1, 5, 3, 6, -2, 5]),
        "frame": np.array([1, 2, 2, 4, 5, 6]),
        "intensity": np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_3d():
    locdata_dict = {
        "position_x": np.array([1, 1, 2, 3, 4, 5]),
        "position_y": np.array([1, 5, 3, 6, 2, 5]),
        "position_z": np.array([1, 2, 5, 4, 3, 2]),
        "frame": np.array([1, 2, 2, 4, 5, 6]),
        "intensity": np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_non_standard_index():
    locdata_dict = {
        "position_x": np.array([1, 1, 2, 3, 4, 5]),
        "position_y": np.array([1, 5, 3, 6, 2, 5]),
        "frame": np.array([1, 2, 2, 4, 5, 6]),
        "intensity": np.array([100, 150, 110, 80, 105, 95]),
    }
    df = pd.DataFrame(locdata_dict)
    df.index = [2, 1, 5, 10, 100, 200]
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    return LocData.from_dataframe(dataframe=df, meta=meta_)


@pytest.fixture(scope="session")
def locdata_rapidSTORM_2d():
    path = Path(ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_data.txt")
    dat = load_rapidSTORM_file(path)
    dat.meta.creation_time.FromSeconds(1)
    return dat


@pytest.fixture(scope="session")
def locdata_blobs_2d():
    path = Path(ROOT_DIR / "tests/test_data/five_blobs.txt")
    dat = load_txt_file(path)
    dat.meta.creation_time.FromSeconds(1)
    return dat


@pytest.fixture(scope="session")
def locdata_blobs_3d():
    path = Path(ROOT_DIR / "tests/test_data/five_blobs_3D.txt")
    dat = load_txt_file(path)
    dat.meta.creation_time.FromSeconds(1)
    return dat


@pytest.fixture(scope="session")
def locdata_two_cluster_2d():
    points = np.array([[0.5, 0.5], [1, 0.6], [1.1, 1], [5, 5.6], [5.1, 6], [5.5, 5]])
    locdata_dict = {
        "position_x": points.T[0],
        "position_y": points.T[1],
        "cluster_label": np.array([1, 1, 1, 2, 2, 2]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    locdata = LocData.from_dataframe(dataframe=df, meta=meta_)
    locdata.region = locdata.bounding_box.region
    return locdata


@pytest.fixture(scope="session")
def locdata_two_cluster_with_noise_2d():
    points = np.array(
        [[0.5, 0.5], [1, 0.6], [1.1, 1], [5, 5.6], [5.1, 6], [5.5, 5], [100, 100]]
    )
    locdata_dict = {
        "position_x": points.T[0],
        "position_y": points.T[1],
        "cluster_label": np.array([1, 1, 1, 2, 2, 2, -1]),
    }
    df = pd.DataFrame(locdata_dict)
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.seconds = 1
    locdata = LocData.from_dataframe(dataframe=df, meta=meta_)
    locdata.region = locdata.bounding_box.region
    return locdata
