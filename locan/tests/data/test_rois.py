import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from locan import ROOT_DIR, Ellipse, LocData, Polygon, Rectangle, Roi, rasterize
from locan.data import metadata_pb2


def test_Roi_2d(locdata_2d):
    roi = Roi(reference=locdata_2d, region=Rectangle((0, 0), 2, 2, 0))
    assert (
        repr(roi) == f"Roi("
        f"reference={locdata_2d}, "
        f"region=Rectangle((0, 0), 2, 2, 0),  "
        f"loc_properties=())"
    )
    assert isinstance(roi._region, Rectangle)
    assert isinstance(roi.region, Rectangle)

    new_dat = roi.locdata()
    assert len(new_dat) == 1
    assert new_dat.region is roi.region

    roi.region = Ellipse((0, 0), 2, 2, 0)
    assert isinstance(roi.region, Ellipse)

    roi = Roi(
        reference=locdata_2d,
        region=Rectangle((0, 0), 4, 4, 0),
        loc_properties=("position_y", "frame"),
    )
    new_dat = roi.locdata()
    assert len(new_dat) == 2
    assert new_dat.region is None

    locdata_empty = LocData()
    roi_empty = Roi(reference=locdata_empty, region=Rectangle((0, 0), 4, 4, 0))
    empty_dat = roi_empty.locdata()
    assert len(empty_dat) == 0
    assert new_dat.region is None

    roi = Roi(
        reference=dict(file_path="test_path", file_type=0),
        region=Rectangle((0, 0), 2, 2, 0),
    )
    # print(roi.reference)
    assert isinstance(roi.reference, metadata_pb2.Metadata)
    assert roi.reference.file.type == 0
    assert roi.reference.file.path == "test_path"

    meta = metadata_pb2.Metadata()
    meta.file.path = "test_path"
    meta.file.type = 0
    roi = Roi(reference=meta, region=Rectangle((0, 0), 2, 2, 0))
    # print(roi.reference)
    assert isinstance(roi.reference, metadata_pb2.Metadata)
    assert roi.reference.file.type == 0
    assert roi.reference.file.path == "test_path"

    meta = metadata_pb2.File()
    meta.path = "test_path"
    meta.type = 0
    roi = Roi(reference=meta, region=Rectangle((0, 0), 2, 2, 0))
    # print(roi.reference)
    assert isinstance(roi.reference, metadata_pb2.Metadata)
    assert roi.reference.file.type == 0
    assert roi.reference.file.path == "test_path"


def test_Roi_3d(locdata_3d):
    roi = Roi(reference=locdata_3d, region=Rectangle((0, 0), 2, 2, 0))
    new_dat = roi.locdata()
    assert len(new_dat) == 1
    assert new_dat.region is None
    assert isinstance(roi.region, Rectangle)
    roi.region = Ellipse((0, 0), 2, 2, 0)
    assert isinstance(roi.region, Ellipse)

    roi = Roi(
        reference=locdata_3d,
        region=Rectangle((0, 0), 2, 3, 0),
        loc_properties=("position_y", "position_z"),
    )
    new_dat = roi.locdata()
    assert len(new_dat) == 1
    assert new_dat.region is None


def test_pickling_locdata_from_Roi(locdata_2d):
    import pickle

    roi = Roi(reference=locdata_2d, region=Rectangle((0, 0), 2, 1, 0))
    new_dat = roi.locdata()

    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "pickled_locdata.pickle"
        with open(file_path, "wb") as file:
            pickle.dump(new_dat, file, pickle.HIGHEST_PROTOCOL)
        with open(file_path, "rb") as file:
            locdata = pickle.load(file)  # noqa S301
        assert len(new_dat) == len(locdata)
        assert isinstance(locdata.meta, metadata_pb2.Metadata)


def test_Roi_io(locdata_2d):
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "roi.yaml"
        # file_path = ROOT_DIR / 'tests/test_data/roi.yaml'

        roi = Roi(reference=locdata_2d, region=Rectangle((0, 0), 2, 1, 10))
        with pytest.warns(UserWarning):
            roi.to_yaml(path=file_path)

        roi = Roi(region=Rectangle((0, 0), 2, 1, 0))
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new.reference is None
        assert isinstance(roi_new.region, Rectangle)
        assert roi_new.loc_properties == []

        roi = Roi(region=Ellipse((0, 0), 2, 1, 0))
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new.reference is None
        assert isinstance(roi_new.region, Ellipse)
        assert roi_new.loc_properties == []

        roi = Roi(region=Polygon(((0, 0), (0, 1), (1, 0.5), (0.8, 0.2))))
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new.reference is None
        assert isinstance(roi_new.region, Polygon)
        assert roi_new.loc_properties == []

        roi = Roi(
            reference=dict(
                file_path=ROOT_DIR / "tests/test_data/five_blobs.txt", file_type=1
            ),
            region=Rectangle((0, 0), 2, 1, 0),
        )
        assert isinstance(roi.reference, (metadata_pb2.Metadata, Path))
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new

        # test region specs with numpy floats
        roi = Roi(
            reference=dict(file_path=file_path, file_type=1),
            region=Rectangle((0, 0), float(2), float(1), float(10)),
        )
        assert isinstance(roi.reference, metadata_pb2.Metadata)
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new

        meta_ = metadata_pb2.Metadata()
        meta_.file.path = str(file_path)
        meta_.file.type = 1
        locdata_2 = LocData.from_selection(locdata_2d, meta=meta_)
        roi = Roi(reference=locdata_2, region=Rectangle((0, 0), 2, 1, 0))
        assert isinstance(roi.reference.meta, metadata_pb2.Metadata)
        roi.to_yaml(path=file_path)

        roi_new = Roi.from_yaml(path=file_path)
        assert roi_new
        assert isinstance(roi_new.reference, metadata_pb2.Metadata)


def test_roi_locdata_from_file(locdata_blobs_3d):
    roi = Roi(reference=locdata_blobs_3d, region=Rectangle((100, 500), 200, 200, 10))
    dat = roi.locdata()
    assert len(dat) == 10

    roi = Roi(
        reference=locdata_blobs_3d,
        region=Rectangle((100, 3), 200, 2, 0),
        loc_properties=["position_x", "cluster_label"],
    )
    dat = roi.locdata()
    assert len(dat) == 10


def test_as_artist():
    roi_rectangle = Roi(reference=None, region=Rectangle((0, 0), 0.5, 0.5, 10))
    roi_ellipse = Roi(reference=None, region=Ellipse((0.5, 0.5), 0.2, 0.3, 10))
    roi_polygon = Roi(
        reference=None,
        region=Polygon(((0, 0.8), (0.1, 0.9), (0.6, 0.9), (0.7, 0.7), (0, 0.8))),
    )
    fig, ax = plt.subplots()
    ax.add_patch(roi_rectangle._region.as_artist())
    ax.add_patch(roi_ellipse._region.as_artist())
    ax.add_patch(roi_polygon._region.as_artist())
    # plt.show()

    plt.close("all")


# to be deprecated
# @pytest.mark.gui
# def test_select_by_drawing_mpl():
#     dat = load_txt_file(path=ROOT_DIR / 'tests/test_data/five_blobs.txt')
#     select_by_drawing_mpl(dat, region_type='rectangle')
#     select_by_drawing_mpl(dat, region_type='ellipse')


# standard LocData fixtures
def test_rasterize(locdata_2d, locdata_single_localization, locdata_non_standard_index):
    res = rasterize(locdata=locdata_2d, support=None, n_regions=(5, 2))
    assert isinstance(res[0].region, Rectangle)
    assert repr(res[0].region) == "Rectangle((1.0, 1.0), 0.8, 2.5, 0)"
    assert len(res) == 10

    res = rasterize(locdata=locdata_2d, support=((0, 10), (10, 20)), n_regions=(5, 2))
    assert isinstance(res[0].region, Rectangle)
    assert repr(res[0].region) == "Rectangle((0.0, 10.0), 2.0, 5.0, 0)"
    assert len(res) == 10

    res = rasterize(
        locdata=locdata_single_localization,
        support=((0, 10), (10, 20)),
        n_regions=(5, 2),
    )
    assert isinstance(res[0].region, Rectangle)
    assert repr(res[0].region) == "Rectangle((0.0, 10.0), 2.0, 5.0, 0)"
    assert len(res) == 10

    res = rasterize(
        locdata=locdata_non_standard_index,
        support=((0, 10), (10, 20)),
        n_regions=(5, 2),
    )
    assert isinstance(res[0].region, Rectangle)
    assert repr(res[0].region) == "Rectangle((0.0, 10.0), 2.0, 5.0, 0)"
    assert len(res) == 10


def test_rasterize_(locdata_empty):
    with pytest.raises(ValueError):
        rasterize(locdata=locdata_empty, support=None, n_regions=(5, 2))
    with pytest.raises(ValueError):
        rasterize(locdata=locdata_empty, support=((0, 10), (10, 20)), n_regions=(5, 2))


def test_rasterize_3d(locdata_3d):
    res = rasterize(
        locdata=locdata_3d,
        support=None,
        n_regions=(5, 2),
        loc_properties=["position_x", "position_z"],
    )
    assert isinstance(res[0].region, Rectangle)
    assert repr(res[0].region) == "Rectangle((1.0, 1.0), 0.8, 2.0, 0)"
    assert res[0].loc_properties == ["position_x", "position_z"]
    assert len(res) == 10
