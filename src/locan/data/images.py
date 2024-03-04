"""
Image class

This module provides an adapter class for Image objects of third-party
image processing libraries.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from google.protobuf import json_format

from locan.data import metadata_pb2
from locan.data.metadata_utils import merge_metadata
from locan.dependencies import HAS_DEPENDENCY, needs_package
from locan.process.aggregate import Bins

if HAS_DEPENDENCY["napari"]:
    import napari


__all__: list[str] = ["Image"]

logger = logging.getLogger()


class ImageProtocol(Protocol):
    data: npt.NDArray[Any]
    is_rgb: bool
    bins: Bins | None
    meta: metadata_pb2.Metadata | None


class ArrayApiObject(Protocol):
    __array_namespace__: str


# def is_array_api_obj(x: Any) -> bool:
#     """Check if input is complient with array API standard."""
#     return hasattr(x, "__array_namespace__")


def is_array_api_obj(x: Any) -> bool:
    """Check if input is complient with array API standard."""
    flag = hasattr(x, "__array_namespace__")
    if flag is False:
        try:
            import array_api_compat  # type: ignore

            flag = array_api_compat.is_array_api_obj(x)
        except ImportError:
            pass
    return flag


class ImageBase(ABC):
    """
    Abstract base class of an adapter class for Image objects.
    """

    @property
    @abstractmethod
    def data(self) -> ArrayApiObject | npt.NDArray[Any] | None:
        pass

    @property
    @abstractmethod
    def is_rgb(self) -> bool:
        pass


T_Image = TypeVar("T_Image", bound="Image")


class Image(ImageBase):
    """
    Adapter class for Image objects.

    The original image object is referenced via self._image.
    An array object that complies with the array API standard
    is referenced in self.data
    https://data-apis.org/array-api/latest/index.html

    All attribute requests are looked up in the following order:
    self, self.data, self._image

    Parameters
    ----------
    image:
        Image class to be adapted.
    is_rgb:
        Whether the image is RGB or RGBA.
        If `False` the image is interpreted as a luminance image.
    meta:
        Metadata about the current dataset.

    Attributes
    ----------
    data: ArrayApiObject | npt.NDArray | None
        Image data as array object following the array API standard.
        If no such array can be provided a numpy.NDArray is returned.
        Can be N dimensional. If the last dimension has length
        3 or 4 it can be interpreted as RGB or RGBA if is_rgb is `True`.
    is_rgb: bool
        Whether the image data in self.data will be interpreted as RGB or RGBA.
        If `False` the image is interpreted as a luminance image.
    bins: Bins | None
        Bins instance carrying pixel coordinates
    meta: locan.data.metadata_pb2.Metadata | None
        Metadata about the current dataset.
    """

    def __init__(
        self,
        image: Any | None,
        is_rgb: bool = False,
        meta: metadata_pb2.Metadata | None = None,
    ) -> None:
        self._image: Any | None = image
        self._is_rgb: bool = is_rgb

        self._data: ArrayApiObject | npt.NDArray[Any] | None = None
        self._bins: Bins | None = None
        self.meta: metadata_pb2.Metadata = metadata_pb2.Metadata()

        # meta
        self.meta.identifier = str(uuid.uuid4())
        self.meta.creation_time.GetCurrentTime()
        self.meta = merge_metadata(metadata=self.meta, other_metadata=meta)

    def __getattr__(self, attr):  # type: ignore
        """All non-adapted calls are passed to the self._data and self._image object"""
        if attr.startswith("__") and attr.endswith(
            "__"
        ):  # this is needed to enable pickling
            raise AttributeError
        try:
            return getattr(self._data, attr)
        except AttributeError:
            return getattr(self._image, attr)

    def __getstate__(self) -> dict[str, Any]:
        """Modify pickling behavior."""
        # Copy the object's state from self.__dict__ to avoid modifying the original state.
        state = self.__dict__.copy()
        # Serialize the unpicklable protobuf entries.
        json_string = json_format.MessageToJson(
            self.meta, including_default_value_fields=False
        )
        state["meta"] = json_string
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Modify pickling behavior."""
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore protobuf class for meta attribute
        self.meta = metadata_pb2.Metadata()
        self.meta = json_format.Parse(state["meta"], self.meta)

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        if is_array_api_obj(value):
            self._data = value
        else:
            self._data = np.asarray(value)
            logger.warning(
                "The data object is not compliant with the array API standard."
            )

    @property
    def is_rgb(self) -> bool:
        return self._is_rgb

    @property
    def bins(self) -> Bins | None:
        return self._bins

    @bins.setter
    def bins(self, bins: Bins) -> None:
        if bins.n_bins == self.shape:
            self._bins = bins
        else:
            raise ValueError("bins and image must have the same shape.")

    @classmethod
    def from_array(
        cls: type[T_Image],  # noqa: UP006
        array: Any,
        is_rgb: bool = False,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        new_image = cls(image=array, is_rgb=is_rgb, meta=meta)
        new_image.data = new_image._image
        return new_image

    @classmethod
    def from_numpy(
        cls: type[T_Image],  # noqa: UP006
        array: npt.ArrayLike,
        is_rgb: bool = False,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        new_image = cls(image=np.asarray(array), is_rgb=is_rgb, meta=meta)
        new_image.data = new_image._image
        return new_image

    @classmethod
    def from_bins(
        cls: type[T_Image],  # noqa: UP006
        bins: Bins,
        value: float | int = 1,
        is_rgb: bool = False,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        data = np.full(fill_value=value, shape=bins.n_bins)
        new_image = cls(image=data, is_rgb=is_rgb, meta=meta)
        new_image.data = new_image._image
        new_image._bins = bins
        return new_image

    @classmethod
    @needs_package("napari")
    def from_napari(
        cls: type[T_Image],  # noqa: UP006
        image: napari.layers.Image | napari.types.ImageData,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        # LayerData = Union[Tuple[DataType], Tuple[DataType, LayerProps], FullLayerData]
        if isinstance(image, tuple):
            image = napari.layers.Layer.create(*image)

        if not isinstance(image, napari.layers.Image):
            raise TypeError("Layer data must be of type Image.")

        new_image = cls(image=image, is_rgb=image.rgb, meta=meta)
        new_image.data = image.data
        return new_image

    @classmethod
    def from_pillow(
        cls: type[T_Image],  # noqa: UP006
        image: Any,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        is_rgb = True if image.mode == "RGB" or image.mode == "RGBA" else False
        new_image = cls(image=image, is_rgb=is_rgb, meta=meta)
        new_image.data = new_image.image
        return new_image
