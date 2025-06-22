"""
Image class

This module provides an adapter class for Image objects of third-party
image processing libraries.

The Image class is the locan way to keep an image, the pixel coordinates
and metadata in sync.

Image data is kept as an array compliant with the Python array API
standard [1]_.

Pixel coordinates are kept as a :class:`Bins` instance.

To create an Image object from any other image library
modify the initialization of self.data and other attributes accordingly:

    class MyImage(Image):
        def __init__(self, image: Any):
            super().__init__(image=image)
            self.data = some_function_or_attribute(self._image)

References
----------
.. [1] https://data-apis.org/array-api/latest

"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from google.protobuf import json_format

from locan.data import metadata_pb2
from locan.data.metadata_utils import merge_metadata
from locan.dependencies import HAS_DEPENDENCY, needs_package

if TYPE_CHECKING:
    from locan.process.aggregate import Bins

if HAS_DEPENDENCY["napari"]:
    import napari


__all__: list[str] = ["Image"]

logger = logging.getLogger()


class ArrayApiObject(Protocol):
    __array_namespace__: str


class ImageProtocol(Protocol):
    """
    Interface specification for an adapter class for Image objects.
    """

    data: ArrayApiObject | npt.NDArray[Any]
    is_rgb: bool
    bins: Bins | None
    meta: metadata_pb2.Metadata


class PillowImage(Protocol):
    """
    Interface specification for a pillow image.
    """

    mode: str
    __array_interface__: dict[str, Any]


def is_array_api_obj(x: Any) -> bool:
    """
    Check if input is compliant with array API standard.
    """
    flag = hasattr(x, "__array_namespace__")
    if flag is False:
        try:
            import array_api_compat  # type: ignore

            flag = array_api_compat.is_array_api_obj(x)
        except ImportError:
            logger.warning(
                "ImportError: Install array_api_compat "
                "to check if is_array_api_obj(x)"
            )
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

    @property
    @abstractmethod
    def bins(self) -> Bins | None:
        pass


T_Image = TypeVar("T_Image", bound="Image")


class Image(ImageBase):
    """
    Adapter class for Image objects.

    The original image object is referenced via self._image.
    An array object that complies with the array API standard
    is referenced in self.data

    All attribute requests are looked up in the following order:
    self, self.data, self._image

    For initiation use the appropriate constructor method.

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
    meta: locan.data.metadata_pb2.Metadata
        Metadata about the current dataset.
    """

    def __init__(
        self,
        image: Any | None = None,
        data: ArrayApiObject | npt.NDArray[Any] | None = None,
        is_rgb: bool = False,
        meta: metadata_pb2.Metadata | None = None,
    ) -> None:
        self._image: Any | None = image
        self._data: ArrayApiObject | npt.NDArray[Any] | None = None
        self._is_rgb: bool = is_rgb
        self._bins: Bins | None = None

        if data is not None:
            self.data = data

        self.meta: metadata_pb2.Metadata = metadata_pb2.Metadata()
        # meta
        self.meta.identifier = str(uuid.uuid4())
        self.meta.creation_time.GetCurrentTime()
        self.meta = merge_metadata(metadata=self.meta, other_metadata=meta)

    def __getattr__(self, attr: str) -> Any:
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
            self.meta, always_print_fields_with_no_presence=False
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
    def data(self) -> ArrayApiObject | npt.NDArray[Any] | None:
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
        """
        Constructor method for any image given as array.

        Parameters
        ----------
        array
            The image data
        is_rgb:
            Whether the image is RGB or RGBA.
            If `False` the image is interpreted as a luminance image.
        meta:
            Metadata about the current dataset.

        Returns
        -------
        Image
        """
        new_image = cls(image=None, data=array, is_rgb=is_rgb, meta=meta)
        return new_image

    @classmethod
    def from_numpy(
        cls: type[T_Image],  # noqa: UP006
        array: npt.ArrayLike,
        is_rgb: bool = False,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        """
        Constructor method for any image given as numpy array.

        Parameters
        ----------
        array
            The image data
        is_rgb:
            Whether the image is RGB or RGBA.
            If `False` the image is interpreted as a luminance image.
        meta:
            Metadata about the current dataset.

        Returns
        -------
        Image
        """
        new_image = cls(image=None, data=np.asarray(array), is_rgb=is_rgb, meta=meta)
        return new_image

    @classmethod
    def from_bins(
        cls: type[T_Image],  # noqa: UP006
        bins: Bins,
        value: float | int = 1,
        is_rgb: bool = False,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        """
        Constructor method for an image with constant values
        and a size that corresponds to the given bin specifications.

        Parameters
        ----------
        bins
            Bin specifications
        value
            A single value as default for the new image.
        is_rgb:
            Whether the image is RGB or RGBA.
            If `False` the image is interpreted as a luminance image.
        meta:
            Metadata about the current dataset.

        Returns
        -------
        Image
        """
        data = np.full(fill_value=value, shape=bins.n_bins)
        new_image = cls(image=None, data=data, is_rgb=is_rgb, meta=meta)
        new_image._bins = bins
        return new_image

    @classmethod
    @needs_package("napari")
    def from_napari(
        cls: type[T_Image],  # noqa: UP006
        image: napari.layers.Image | napari.types.ImageData,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        """
        Constructor method for an image derived from a napari.Image instance.

        Parameters
        ----------
        image
            The napari image or image data.
            Image can be of type LayerData for Image, i.e.,
            Union[Tuple[DataType], Tuple[DataType, LayerProps], FullLayerData]
        meta:
            Metadata about the current dataset.

        Returns
        -------
        Image
        """
        if isinstance(image, tuple):
            image = napari.layers.Layer.create(*image)

        if not isinstance(image, napari.layers.Image):
            raise TypeError("Layer data must be of type Image.")

        new_image = cls(image=image, data=image.data, is_rgb=image.rgb, meta=meta)
        return new_image

    @classmethod
    def from_pillow(
        cls: type[T_Image],  # noqa: UP006
        image: PillowImage,
        meta: metadata_pb2.Metadata | None = None,
    ) -> T_Image:
        """
        Constructor method for an image derived from a pillow.Image instance.

        Parameters
        ----------
        image
            The pillow image.
        meta:
            Metadata about the current dataset.

        Returns
        -------
        Image
        """
        is_rgb = True if image.mode == "RGB" or image.mode == "RGBA" else False
        new_image = cls(image=image, data=np.array(image), is_rgb=is_rgb, meta=meta)
        return new_image
