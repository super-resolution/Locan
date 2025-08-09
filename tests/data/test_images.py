import pickle
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy import array_equal

from locan import Bins, Image
from locan.data import metadata_pb2
from locan.dependencies import HAS_DEPENDENCY

napari = pytest.importorskip("napari")

HAS_NAPARI_AND_PYTESTQT = HAS_DEPENDENCY["napari"] and HAS_DEPENDENCY["pytestqt"]
# pytestqt is not a requested or extra dependency.
# If napari and pytest-qt is installed, all tests run.
# Tests in docker or GitHub actions on linux require xvfb
# for tests with pytest-qt to run.


class TestImage:

    def test_image_init(self):
        image = Image()
        assert image._image is None
        assert image.data is None
        assert image.is_rgb is False
        assert image.bins is None
        assert isinstance(image.meta, metadata_pb2.Metadata)
        with pytest.raises(AttributeError):
            assert image.shape

        image = Image(image=np.zeros(shape=(2, 3)))
        assert image._image.shape == (2, 3)
        assert image.data is None
        assert image.is_rgb is False
        assert image.bins is None
        assert isinstance(image.meta, metadata_pb2.Metadata)
        assert image.shape == (2, 3)
        assert image.ndim == 2

        image.meta.file.path = "some/path"
        assert image.meta.file.path == "some/path"
        assert image.meta.creation_time != ""

        bins = Bins(n_bins=(2, 3), bin_range=(10, 100))
        image.bins = bins
        assert image.bins.n_bins == image.shape

        bins = Bins(n_bins=(2, 4), bin_range=(10, 100))
        with pytest.raises(ValueError):
            image.bins = bins

        image = Image(
            image=np.zeros(shape=(2, 3)), is_rgb=True, meta={"identifier": "1"}
        )
        assert image._image.shape == (2, 3)
        assert image.data is None
        assert image.is_rgb is True
        assert image.bins is None
        assert isinstance(image.meta, metadata_pb2.Metadata)
        assert image.shape == (2, 3)
        assert image.ndim == 2
        assert image.meta.identifier == "1"

        image = Image.from_array(array=np.zeros(shape=(2, 3)), meta={"identifier": "1"})
        assert image._image is None
        assert image.data.shape == (2, 3)
        assert image.is_rgb is False
        assert image.bins is None
        assert image.shape == (2, 3)
        assert image.meta.identifier == "1"

    def test_constructor_methods(self):
        image = Image.from_numpy(array=np.zeros(shape=(2, 3)), meta={"identifier": "1"})
        assert image._image is None
        assert image.data.shape == (2, 3)
        assert image.is_rgb is False
        assert image.bins is None
        assert image.shape == (2, 3)
        assert image.meta.identifier == "1"

        bins = Bins(n_bins=(2, 3), bin_range=(10, 100))
        image = Image.from_bins(bins=bins, meta={"identifier": "1"})
        assert image._image is None
        assert image.data.shape == (2, 3)
        assert image.is_rgb is False
        assert image.bins == bins
        assert image.shape == (2, 3)
        assert image.data[0, 0] == 1
        assert image.meta.identifier == "1"

        bins = Bins(n_bins=(2, 3), bin_range=(10, 100))
        image = Image.from_bins(bins=bins, value=2)
        assert image._image is None
        assert image.data.shape == (2, 3)
        assert image.shape == (2, 3)
        assert image.data[0, 0] == 2
        image.data = np.array([[1, 2, 3], [4, 5, 6]])
        assert image.data[0, 0] == 1

    def test_image_from_pillow(self):
        class PillowMock:
            def __init__(self, mode, data):
                self.mode = mode
                self._data = data

            @property
            def __array_interface__(self):
                array_interface = {
                    "shape": self._data.shape,
                    "typestr": "i",
                    "data": self._data,
                    "version": 3,
                }
                return array_interface

        pillow_image = PillowMock(mode="RGBA", data=np.zeros(shape=(2, 3, 4)))
        image = Image.from_pillow(image=pillow_image, meta={"identifier": "1"})
        assert image._image is pillow_image
        assert image.data.shape == (2, 3, 4)
        assert image.is_rgb is True
        assert image.bins is None
        assert image.shape == (2, 3, 4)
        assert image.meta.identifier == "1"

        pillow_image = PillowMock(mode="L", data=np.zeros(shape=(2, 3)))
        image = Image.from_pillow(image=pillow_image, meta={"identifier": "1"})
        assert image._image is pillow_image
        assert image.data.shape == (2, 3)
        assert image.is_rgb is False
        assert image.bins is None
        assert image.shape == (2, 3)
        assert image.meta.identifier == "1"

    @pytest.mark.skipif(
        not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
    )
    def test_image_from_napari(self):
        image_in = napari.Viewer().add_image(data=np.zeros(shape=(2, 3)))
        image = Image.from_napari(image=image_in)
        assert image._image.dtype == float
        assert image.data.shape == (2, 3)
        assert image.is_rgb is False
        assert image.bins is None
        assert image.shape == (2, 3)
        assert isinstance(image.meta, metadata_pb2.Metadata)

        image_in = napari.Viewer().add_image(data=np.zeros(shape=(2, 3, 4)), rgb=True)
        image = Image.from_napari(image=image_in, meta={"identifier": "1"})
        assert image._image.dtype == float
        assert image.data.shape == (2, 3, 4)
        assert image.is_rgb is True
        assert image.bins is None
        assert image.shape == (2, 3, 4)
        assert image.meta.identifier == "1"

        image_in = napari.Viewer().add_image(data=np.zeros(shape=(2, 3)))
        image_in = image_in.as_layer_data_tuple()
        image = Image.from_napari(image=image_in)
        assert image._image.dtype == float
        assert image.data.shape == (2, 3)
        assert image.is_rgb is False
        assert image.bins is None
        assert image.shape == (2, 3)
        assert isinstance(image.meta, metadata_pb2.Metadata)

    def test_plugin(self):
        """
        To create an Image object from any other library object
        modify the initialization of self.data and other attributes accordingly.
        """

        class ImageMock:
            def __init__(self, data):
                self._data = data

        class MyImage(Image):
            def __init__(self, image: ImageMock, meta=None):
                super().__init__(image=image, meta=meta)
                self._image = cast(ImageMock, self._image)
                self.data = self._image._data

        third_party_image = ImageMock(np.zeros(shape=(2, 3)))
        image = MyImage(image=third_party_image, meta={"identifier": "1"})
        assert image.data.shape == (2, 3)
        assert image.shape == (2, 3)
        assert image.is_rgb is False
        assert image.bins is None
        assert image.meta.identifier == "1"

        bins = Bins(n_bins=(2, 3), bin_range=(10, 100))
        image.bins = bins
        assert image.bins.n_bins == image.shape

    def test_image_pickling(self):
        image = Image.from_array(
            array=np.zeros(shape=(2, 3)), is_rgb=True, meta={"identifier": "1"}
        )
        assert image.is_rgb is True
        assert image.meta.identifier == "1"
        assert array_equal(image.data, np.zeros(shape=(2, 3)))

        with tempfile.TemporaryDirectory() as tmp_directory:
            file_path = Path(tmp_directory) / "pickled_image.pickle"
            with open(file_path, "wb") as file:
                pickle.dump(image, file, pickle.HIGHEST_PROTOCOL)
            with open(file_path, "rb") as file:
                picked_image = pickle.load(file)  # noqa S301

            assert array_equal(picked_image.data, np.zeros(shape=(2, 3)))
            assert picked_image.is_rgb is True
            assert isinstance(picked_image.meta, metadata_pb2.Metadata)
            assert picked_image.meta.identifier == "1"
