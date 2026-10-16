"""

Fourier shell correlation (FSC) analysis.

A resolution measure based on comparison of two image representations
that takes labeling density and localization precision into account.
In FSC a resolution is determined for a 3-dimensional image.

See Also
---------
:class:`FourierRingCorrelation`

References
----------
.. [1] Nieuwenhuizen, R., Lidke, K., Bates, M. et al.,
       MeasuShell image resolution in optical nanoscopy.
       Nat Methods 10, 557–562 (2013),
       https://doi.org/10.1038/nmeth.2448
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import scipy.signal.windows
from scipy.ndimage import gaussian_filter1d

from locan import Bins, LocData, histogram
from locan.process.properties.locdata_statistics import range_from_collection, ranges

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from locan.analysis import metadata_analysis_pb2
from locan.analysis.analysis_base import _Analysis
from locan.analysis.fourier_ring_correlation import get_frequency_at_threshold
from locan.data.images import Image

__all__: list[str] = ["FourierShellCorrelation", "FourierShellCorrelationResults"]

logger = logging.getLogger(__name__)


# The algorithms


class FscReturn(NamedTuple):
    """Return value for func:`compute_fourier_shell_correlation`."""

    frequency: npt.NDArray[np.float64]
    correlation: npt.NDArray[np.float64]


def preprocess_images(
    image: npt.ArrayLike,
    other_image: npt.ArrayLike,
    window: str | float | tuple | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Prepare 3-dimensional image data for computing fourier shell correlation.
    Adjust shape of smaller image py padding and apply windowing.

    Parameters
    ----------
    image:
        First image representation.
    other_image:
        Second image representation.
    window:
        If not None, apply a window function to the generated images.
        Typically, a Tukey filter is applied: window=("tukey", 0.75)
        The window is specified according to func:`scipy.signal.get_window`.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    """

    image = np.asarray(image)
    other_image = np.asarray(other_image)

    if image.shape == other_image.shape:
        shape: tuple[int, int, int] = image.shape
    else:
        logger.warning("Shape of image and other_image is different.")
        shape = tuple(
            max(item, other_item)
            for item, other_item in zip(image.shape, other_image.shape)
        )

        pad_width_0, pad_width_1, pad_width_2 = tuple(
            item - other_item for item, other_item in zip(shape, image.shape)
        )
        image = np.pad(
            image,
            pad_width=((0, pad_width_0), (0, pad_width_1), (0, pad_width_2)),
            mode="constant",
            constant_values=0,
        )

        pad_width_0, pad_width_1, pad_width_2 = tuple(
            item - other_item for item, other_item in zip(shape, other_image.shape)
        )
        other_image = np.pad(
            other_image,
            pad_width=((0, pad_width_0), (0, pad_width_1), (0, pad_width_2)),
            mode="constant",
            constant_values=0,
        )

    # apply windowing
    if window is not None:
        window_x = scipy.signal.windows.get_window(window=window, Nx=shape[0])
        window_y = scipy.signal.windows.get_window(window=window, Nx=shape[1])
        window_z = scipy.signal.windows.get_window(window=window, Nx=shape[2])
        window_ = (
            window_x[:, None, None] * window_y[None, :, None] * window_z[None, None, :]
        )

        image = image * window_
        other_image = other_image * window_

    return image, other_image


def compute_fourier_shell_correlation(
    image: npt.ArrayLike, other_image: npt.ArrayLike, bin_size: float
) -> FscReturn:
    """
    Compute Fourier shell correlation from two image representations.

    Parameters
    ----------
    image:
        First image representation.
    other_image:
        Second image representation.
    bin_size:
        Pixel size that must be identical for both images and all dimensions.

    Returns
    -------
    FscReturn
    """
    image = np.asarray(image)
    other_image = np.asarray(other_image)

    if image.ndim != 3 or other_image.ndim != 3:
        raise ValueError("Fourier shell correlation requires 3D images.")

    shape = image.shape

    # compute fourier transform:
    F1 = np.fft.fftshift(np.fft.fftn(image))
    F2 = np.fft.fftshift(np.fft.fftn(other_image))

    # create image with values reflecting the distance from pixel to image center (r) in units of pixel
    nz, ny, nx = shape
    z = np.arange(nz) - nz // 2
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)

    max_r = int(min(shape) / 2)

    # compute fourier shell correlation for each distance
    fsc = []
    frequencies = []
    for shell in range(1, max_r):
        mask = (r >= shell) & (r < shell + 1)

        # select pixel reflecting the right distance
        a = F1[mask]
        b = F2[mask]

        # complex product
        num = np.sum(a * np.conj(b))

        # normalization
        den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))

        # real part of ratio is the FSC
        fsc.append(np.real(num / den))
        frequencies.append(shell / (shape[0] * bin_size))

    return FscReturn(frequency=np.asarray(frequencies), correlation=np.asarray(fsc))


# The specific analysis classes


@dataclass
class FourierShellCorrelationResults:
    frequency: npt.NDArray[np.float64] = field(repr=False)
    correlation: npt.NDArray[np.float64] = field(repr=False)
    image: Image = field(repr=False)
    other_image: Image = field(repr=False)

    def threshold_frequency(self, threshold: float = 1 / 7) -> float:
        """
        Get the frequency at which the correlation drops below the threshold value.
        As default, we use a threshold of 1 / 7 = ~0.143 that is established in the field.

        Parameters
        ----------
        threshold
            The correlation threshold

        Returns
        -------
        float
        """
        return get_frequency_at_threshold(
            frequencies=self.frequency,
            correlation=self.correlation,
            threshold=threshold,
        )

    def resolution(self, threshold: float = 1 / 7) -> float:
        """
        Get the resolution = 1/threshold_frequency at which the correlation drops below the threshold value.
        As default, we use a threshold of 1 / 7 = ~0.143 that is established in the field.

        Parameters
        ----------
        threshold
            The correlation threshold

        Returns
        -------
        float
        """
        return 1 / self.threshold_frequency(threshold=threshold)


class FourierShellCorrelation(_Analysis):
    """
    Compute the fourier shell correlation for data or
    between data and other_data.

    Parameters
    ----------
    meta
        Metadata about the current analysis routine.
    bin_size:
        The pixel size for the computed image.
    smooth:
        If not False, apply a Gaussian filter on the correlation with sigma=smooth.
    window:
        If not None, apply a window function to the generated images.
        Typically, a Tukey filter is applied: window=("tukey", 0.75)
        The window is specified according to func:`scipy.signal.get_window`.
    Attributes
    ----------
    count : int
        A counter for counting instantiations.
    parameter : dict
        A dictionary with all settings for the current computation.
    meta : locan.analysis.metadata_analysis_pb2.AMetadata
        Metadata about the current analysis routine.
    results : FourierShellCorrelationResults
        Computed results.
    """

    count = 0

    def __init__(
        self,
        meta: metadata_analysis_pb2.AMetadata | None = None,
        bin_size: float = 10,
        smooth: int | float | Literal[False] = False,
        window: str | float | tuple | None = None,
    ) -> None:
        parameters = self._get_parameters(locals())
        super().__init__(**parameters)
        self.results: FourierShellCorrelationResults | None = None

    def compute(self, locdata: LocData, other_locdata: LocData | None = None) -> Self:
        """
        Run the computation.

        Parameters
        ----------
        locdata
           Localization data.
        other_locdata
            Other localization data to be taken as other image.

        Returns
        -------
        Self
        """
        if not len(locdata):
            logger.warning("Locdata is empty.")
            return self

        # create two image representations
        if other_locdata is None:
            locdata_chunks = LocData.from_chunks(
                locdata=locdata, n_chunks=2, order="alternating"
            )

            bin_range = ranges(locdata=locdata)
            assert bin_range is not None  # type narrowing # noqa: S101
            # pyrefly: ignore [bad-argument-type]
            bins = Bins(bin_range=bin_range, bin_size=self.parameter["bin_size"])

            assert isinstance(  # type narrowing # noqa: S101
                locdata_chunks.references, Iterable
            )
            image_data, other_image_data = tuple(
                histogram(locdata=locdata_, bins=bins).data
                for locdata_ in locdata_chunks.references
            )

        else:
            if not len(other_locdata):
                logger.warning("Other locdata is empty.")
                return self

            # check dimensions
            if locdata.dimension != other_locdata.dimension:
                raise TypeError(
                    "Dimensions for locdata and other_locdata must be identical."
                )

            bin_range = range_from_collection(locdatas=[locdata, other_locdata])
            bins = Bins(bin_range=bin_range, bin_size=self.parameter["bin_size"])
            image_data, other_image_data = tuple(
                histogram(locdata=locdata_, bins=bins).data
                for locdata_ in [locdata, other_locdata]
            )

        # process data
        image_data, other_image_data = preprocess_images(
            image_data, other_image_data, window=self.parameter["window"]
        )
        fsc_result = compute_fourier_shell_correlation(
            image=image_data,
            other_image=other_image_data,
            bin_size=self.parameter["bin_size"],
        )

        if self.parameter["smooth"] is not False:
            fsc_values = gaussian_filter1d(
                fsc_result.correlation, sigma=self.parameter["smooth"]
            )
        else:
            fsc_values = fsc_result.correlation

        image = Image.from_numpy(image_data)
        image.bins = bins
        other_image = Image.from_numpy(other_image_data)
        other_image.bins = bins

        self.results = FourierShellCorrelationResults(
            frequency=fsc_result.frequency,
            correlation=fsc_values,
            image=image,
            other_image=other_image,
        )

        return self

    def plot(
        self,
        ax: mpl.axes.Axes | None = None,
        threshold: float | None = 1 / 7,
        **kwargs: Any,
    ) -> mpl.axes.Axes:
        """
        Provide plot showing the fourier shell correlation.

        Parameters
        ----------
        ax
            The axes on which to show the image
        kwargs
            Other parameters passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        if self.results is None:
            return ax

        ax.plot(self.results.frequency, self.results.correlation, **kwargs)
        ax.set(
            title="Fourier Shell Correlation",
            xlabel="Spatial frequency",
            ylabel="FSC",
            ylim=(-0.01, 1.01),
        )
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--")
            ax.axvline(
                self.results.threshold_frequency(threshold=threshold),
                color="red",
                linestyle="--",
            )
            ax.text(
                0.6,
                0.9,
                f"Resolution: {self.results.resolution(threshold=threshold):0.0f}",
                transform=ax.transAxes,
            )

        return ax
