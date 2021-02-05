from pathlib import Path
import tempfile

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.rois import Roi, rasterize
from surepy.render.render2d import select_by_drawing_mpl, _napari_shape_to_RoiRegion, select_by_drawing_napari
from surepy.data import metadata_pb2


# fixtures

@pytest.fixture()
def locdata():
    locdata_dict = {
        'position_x': [0, 1, 2, 3, 0, 1, 4, 5],
        'position_y': [0, 1, 2, 3, 1, 4, 5, 1],
        'position_z': [0, 1, 2, 3, 4, 4, 4, 5]
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def points():
    return np.array([[0, 0], [0.5, 0.5], [100, 100], [1, 1], [1.1, 0.9]])


# tests

def test_Roi(locdata):
    roi = Roi(reference=locdata, region_specs=((0, 0), 2, 1, 0), region_type='rectangle')
    new_dat = roi.locdata()
    assert len(new_dat) == 2

    roi = Roi(reference=locdata, region_specs=((0, 0), 2, 3, 0), region_type='rectangle',
              properties_for_roi=('position_y', 'position_z'))
    new_dat = roi.locdata()
    assert len(new_dat) == 2

    locdata_empty = LocData()
    roi_empty = Roi(reference=locdata_empty, region_specs=((0, 0), 2, 1, 10), region_type='rectangle')
    empty_dat = roi_empty.locdata()
    assert len(empty_dat) == 0


def test_Roi_io(locdata):
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / 'roi.yaml'
        #file_path = ROOT_DIR / 'tests/test_data/roi.yaml'

        roi = Roi(region_specs=((0, 0), 2, 1, 10), region_type='rectangle')
        roi.to_yaml(path=file_path)

        roi = Roi(reference=locdata, region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
        with pytest.warns(UserWarning):
            roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new.reference is None

        roi = Roi(reference=dict(file_path=ROOT_DIR / 'tests/test_data/five_blobs.txt', file_type=1),
                  region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
        assert isinstance(roi.reference, (metadata_pb2.Metadata, Path))
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new

        # test region specs with numpy floats
        roi = Roi(reference=dict(file_path=file_path, file_type=1),
                  region_type='rectangle',
                  region_specs=(np.array([0, 0], dtype=np.float), float(2), float(1), float(10))
                  )
        assert isinstance(roi.reference, metadata_pb2.Metadata)
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new

        locdata_2 = LocData.from_selection(locdata,
                                           meta=dict(file_path=str(file_path),
                                                     file_type=1))
        roi = Roi(reference=locdata_2,
                  region_type='rectangle', region_specs=((0, 0), 2, 1, 10))
        assert isinstance(roi.reference.meta, metadata_pb2.Metadata)
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new
        assert isinstance(roi_new.reference, metadata_pb2.Metadata)


def test_roi_locdata_from_file():
    locdata = load_txt_file(path=ROOT_DIR / 'tests/test_data/five_blobs_3D.txt')

    roi = Roi(reference=locdata, region_type='rectangle', region_specs=((100, 500), 200, 200, 10))
    dat = roi.locdata()
    assert(len(dat) == 10)

    roi = Roi(reference=locdata, region_type='rectangle', region_specs=((100, 100), 200, 200, 10),
              properties_for_roi=['position_x', 'position_z'])
    dat = roi.locdata()
    assert(len(dat) == 10)


def test_as_artist():
    roi_rectangle = Roi(reference=locdata, region_specs=((0, 0), 0.5, 0.5, 10), region_type='rectangle')
    roi_ellipse = Roi(reference=locdata, region_specs=((0.5, 0.5), 0.2, 0.3, 10), region_type='ellipse')
    roi_polygon = Roi(reference=locdata, region_specs=((0, 0.8), (0.1, 0.9), (0.6, 0.9), (0.7, 0.7), (0, 0.8)),
                      region_type='polygon')
    fig, ax = plt.subplots()
    ax.add_patch(roi_rectangle._region.as_artist())
    ax.add_patch(roi_ellipse._region.as_artist())
    ax.add_patch(roi_polygon._region.as_artist())
    # plt.show()


@pytest.mark.skip('GUI tests are skipped because they need user interaction.')
def test_select_by_drawing_mpl():
    dat = load_txt_file(path=ROOT_DIR / 'tests/test_data/five_blobs.txt')
    select_by_drawing_mpl(dat, region_type='rectangle')
    select_by_drawing_mpl(dat, region_type='ellipse')


@pytest.mark.skip('GUI tests are skipped because they need user interaction.')
def test_select_by_drawing_napari():
    dat = load_txt_file(path=ROOT_DIR / 'tests/test_data/five_blobs.txt')
    roi_list = select_by_drawing_napari(dat)
    print(roi_list)


# standard LocData fixtures
def test_rasterize(locdata_2d, locdata_single_localization, locdata_non_standard_index):
    res = rasterize(locdata=locdata_2d, support=None, n_regions=(5, 2))
    assert res[0]._region.region_type == 'rectangle'
    assert repr(res[0]._region.region_specs) == 'RegionSpecs(corner=(1.0, 1.0), width=0.8, height=2.5, angle=0)'
    assert len(res) == 10

    res = rasterize(locdata=locdata_2d, support=((0, 10), (10, 20)), n_regions=(5, 2))
    assert res[0]._region.region_type == 'rectangle'
    assert repr(res[0]._region.region_specs) == 'RegionSpecs(corner=(0.0, 10.0), width=2.0, height=5.0, angle=0)'
    assert len(res) == 10

    res = rasterize(locdata=locdata_single_localization, support=((0, 10), (10, 20)), n_regions=(5, 2))
    assert res[0]._region.region_type == 'rectangle'
    assert repr(res[0]._region.region_specs) == 'RegionSpecs(corner=(0.0, 10.0), width=2.0, height=5.0, angle=0)'
    assert len(res) == 10

    res = rasterize(locdata=locdata_non_standard_index, support=((0, 10), (10, 20)), n_regions=(5, 2))
    assert res[0]._region.region_type == 'rectangle'
    assert repr(res[0]._region.region_specs) == 'RegionSpecs(corner=(0.0, 10.0), width=2.0, height=5.0, angle=0)'
    assert len(res) == 10


def test_rasterize_(locdata_empty):
    with pytest.raises(ValueError):
        rasterize(locdata=locdata_empty, support=None, n_regions=(5, 2))
    with pytest.raises(ValueError):
        rasterize(locdata=locdata_empty, support=((0, 10), (10, 20)), n_regions=(5, 2))


def test_rasterize_3d(locdata_3d):
    res = rasterize(locdata=locdata_3d, support=None, n_regions=(5, 2), properties_for_roi=['position_x', 'position_z'])
    assert res[0]._region.region_type == 'rectangle'
    assert res[0].properties_for_roi == ['position_x', 'position_z']
    assert repr(res[0]._region.region_specs) == 'RegionSpecs(corner=(1.0, 1.0), width=0.8, height=2.0, angle=0)'
    assert len(res) == 10


def test__napari_shape_to_RoiRegion():

    # rectangle
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_RoiRegion(vertices, bin_edges, 'rectangle')
    assert np.array_equal(region.region_specs, np.array(((0.0, 2.0), 25.0, 3.0999999999999996, 0), dtype=object))

    # ellipse
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_RoiRegion(vertices, bin_edges, 'ellipse')
    assert np.array_equal(region.region_specs, np.array(((12.5, 3.55), 25.0, 3.0999999999999996, 0), dtype=object))

    # polygon
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_RoiRegion(vertices, bin_edges, 'polygon')
    assert np.array_equal(region.region_specs, np.array([[0, 2], [25, 2], [25, 5.1], [0, 5.1], [0, 2]]))
