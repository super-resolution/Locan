"""

This module provides functions to interact with napari
for rendering locdata objects in 2D.

"""
from __future__ import annotations

import logging
from collections.abc import Iterable  # noqa: F401
from typing import Callable  # noqa: F401

import numpy as np
from matplotlib import colors as mcolors

from locan import locdata_id
from locan.data import LocData
from locan.data.aggregate import Bins, histogram
from locan.data.locdata_utils import _check_loc_properties
from locan.dependencies import HAS_DEPENDENCY, needs_package
from locan.visualize.transform import adjust_contrast

if HAS_DEPENDENCY["napari"]:
    import napari

__all__: list[str] = [
    "render_2d_napari",
    "render_2d_napari_image",
    "render_2d_rgb_napari",
]

logger = logging.getLogger(__name__)


@needs_package("napari")
def render_2d_napari_image(
    locdata,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=None,
    bin_edges=None,
    bin_range=None,
    rescale=None,
    viewer=None,
    cmap="viridis",
    **kwargs,
) -> napari.types.LayerData:
    """
    Render localization data into a 2D image by binning x,y-coordinates into
    regular bins. Provide layer data for napari.

    Parameters
    ----------
    locdata : LocData
        Localization data.
    loc_properties : list[str] | None
        Localization properties to be grouped into bins.
        If None The coordinate_values of `locdata` are used.
    other_property : str | None
        Localization property (columns in `locdata.data`) that is averaged
        in each pixel.
        If None, localization counts are shown.
    bins : Bins | boost_histogram.axis.Axis | boost_histogram.axis.AxesTuple | None
        The bin specification as defined in :class:`Bins`
    bin_edges : Sequence[float] | Sequence[Sequence[float]] | None
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]] | str | None
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    n_bins : int | Sequence[int] | None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float | Sequence[float] | Sequence[Sequence[float]] | None
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
        To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
    rescale : int | str | locan.Trafo | Callable | bool | None
        Transformation as defined in :class:`locan.Trafo` or by
        transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min
        intensities.
        For 'equal' intensity values are rescaled by histogram equalization.
    cmap : str | Colormap
        The Colormap object used to map normalized data values to RGBA colors.
    kwargs : dict
        Other parameters passed to :func:`napari.Viewer.add_image`.

    Returns
    -------
    napari.types.LayerData
        Tuple with data, image_kwargs, "image"
    """
    # raise if no or single point in locdata
    if len(locdata) < 2:
        if len(locdata) == 1:
            logger.warning("Locdata carries a single localization.")
        raise ValueError(
            "Locdata has zero or one localizations - must have more than one."
        )

    data, bins, labels = histogram(
        locdata=locdata,
        loc_properties=loc_properties,
        other_property=other_property,
        bins=bins,
        n_bins=n_bins,
        bin_size=bin_size,
        bin_edges=bin_edges,
        bin_range=bin_range,
    )
    if not all(bins.is_equally_sized):
        raise ValueError("All bins must be equally sized.")
    data = adjust_contrast(data, rescale)

    add_image_kwargs = dict(
        name=f"LocData {locdata_id}",
        colormap=cmap,
        scale=bins.bin_size,
        translate=np.asarray(bins.bin_range)[:, 0],
        metadata=dict(message=locdata.meta.SerializeToString()),
    )

    layer_data = (data, dict(add_image_kwargs, **kwargs), "image")
    return layer_data


@needs_package("napari")
def render_2d_napari(
    locdata,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=None,
    bin_edges=None,
    bin_range=None,
    rescale=None,
    viewer=None,
    cmap="viridis",
    **kwargs,
) -> napari.Viewer:
    """
    Render localization data into a 2D image by binning x,y-coordinates into
    regular bins. Render the data using napari.

    Parameters
    ----------
    locdata : LocData
        Localization data.
    loc_properties : list[str] | None
        Localization properties to be grouped into bins.
        If None The coordinate_values of `locdata` are used.
    other_property : str | None
        Localization property (columns in `locdata.data`) that is averaged
        in each pixel.
        If None, localization counts are shown.
    bins : Bins | boost_histogram.axis.Axis | boost_histogram.axis.AxesTuple | None
        The bin specification as defined in :class:`Bins`
    bin_edges : Sequence[float] | Sequence[Sequence[float]] | None
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]] | str | None
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    n_bins : int | Sequence[int] | None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float | Sequence[float] | Sequence[Sequence[float]] | None
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
        To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
    rescale : int | str | locan.Trafo | Callable | bool | None
        Transformation as defined in :class:`locan.Trafo` or by
        transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min
        intensities.
        For 'equal' intensity values are rescaled by histogram equalization.
    viewer : napari.Viewer
        The viewer object on which to add the image
    cmap : str | Colormap
        The Colormap object used to map normalized data values to RGBA colors.
    kwargs : dict
        Other parameters passed to :func:`napari.Viewer.add_image`.

    Returns
    -------
    napari.Viewer
    """
    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer()

    try:
        data, image_kwargs, layer_type = render_2d_napari_image(
            locdata=locdata,
            loc_properties=loc_properties,
            other_property=other_property,
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range,
            rescale=rescale,
            cmap=cmap,
            **kwargs,
        )
        viewer.add_image(data=data, **dict(image_kwargs, **kwargs))

        # set scale_bar unit
        names = list(
            set(
                item.unit
                for item in locdata.meta.localization_properties
                if item.name in locdata.coordinate_keys
            )
        )
        if len(names) == 1:
            viewer.scale_bar.unit = f"1 {names[0]}"
        else:
            viewer.scale_bar.unit = None
    except ValueError as e:
        if (
            len(e.args) > 0
            and e.args[0]
            == "Locdata has zero or one localizations - must have more than one."
        ):
            pass
        else:
            raise e
    return viewer


@needs_package("napari")
def render_2d_rgb_napari(
    locdatas,
    loc_properties=None,
    other_property=None,
    bins=None,
    n_bins=None,
    bin_size=10,
    bin_edges=None,
    bin_range=None,
    rescale=None,
    viewer=None,
    **kwargs,
) -> tuple[napari.Viewer, Bins]:
    """
    Render localization data into a 2D RGB image by binning x,y-coordinates
    into regular bins.

    Note
    ----
    For rescale=False no normalization is carried out image intensities are
    clipped to (0, 1) for float value or (0, 255) for integer values according
    to the matplotlib.imshow behavior.
    For rescale=None we apply a normalization to (min, max) of all intensity
    values.
    For all other rescale options the normalization is applied to each
    individual image.

    Parameters
    ----------
    locdatas : Iterable[LocData]
        Localization data.
    loc_properties : list | None
        Localization properties to be grouped into bins. If None
        The coordinate_values of `locdata` are used.
    other_property : str  | None
        Localization property (columns in `locdata.data`) that is averaged in
        each pixel.
        If None, localization counts are shown.
    bins : Bins | boost_histogram.axis.Axis | boost_histogram.axis.AxesTuple | None
        The bin specification as defined in :class:`Bins`
    bin_edges : Sequence[float] | Sequence[Sequence[float]] | None
        Bin edges for all or each dimension
        with shape (dimension, n_bin_edges).
    bin_range : tuple[float, float] | Sequence[float] | Sequence[Sequence[float]] | str | None
        Minimum and maximum edge for all or each dimensions
        with shape (2,) or (dimension, 2).
        If None (min, max) ranges are determined from data and returned;
        if 'zero' (0, max) ranges with max determined from data are returned.
        if 'link' (min_all, max_all) ranges with min and max determined from
        all combined data are returned.
    n_bins : int | Sequence[int] | None
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size : float | Sequence[float] | Sequence[Sequence[float]] | None
        The size of bins for all or each bin and for all or each dimension
        with shape (dimension,) or (dimension, n_bins).
        5 would describe bin_size of 5 for all bins in all dimensions.
        ((2, 5),) yield bins of size (2, 5) for one dimension.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        ((2, 5), (1, 3)) yields bins of size (2, 5) for one dimension and
        (1, 3) for the other dimension.
        To specify arbitrary sequence of `bin_size` use `bin_edges` instead.
    rescale : int | str | locan.Trafo | Callable | bool | None
        Transformation as defined in :class:`locan.Trafo` or by
        transformation function.
        For None or False no rescaling occurs.
        Legacy behavior:
        For tuple with upper and lower bounds provided in percent,
        rescale intensity values to be within percentile of max and min
        intensities.
        For 'equal' intensity values are rescaled by histogram equalization.
    viewer : napari.Viewer
        The viewer object on which to add the image
    cmap : str | Colormap
        The Colormap object used to map normalized data values to RGBA colors.
    kwargs : dict
        Other parameters passed to :func:`napari.Viewer.add_image`.

    Returns
    -------
    tuple[napari.Viewer, Bins]
    """
    # Provide napari viewer if not provided
    if viewer is None:
        viewer = napari.Viewer()

    locdata_temp = LocData.concat(locdatas)

    # return viewer if no or single point in locdata
    if len(locdata_temp) < 2:
        if len(locdata_temp) == 1:
            logger.warning("Locdata carries a single localization.")
        return viewer

    if bin_edges is None:
        _, bins, labels = histogram(
            locdata=locdata_temp,
            loc_properties=loc_properties,
            other_property=other_property,
            bins=bins,
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range,
        )

    else:
        labels = _check_loc_properties(locdata_temp, loc_properties)
        bins = Bins(bin_edges=bin_edges, labels=labels)

    imgs = [
        histogram(
            locdata=locdata,
            loc_properties=loc_properties,
            other_property=other_property,
            bin_edges=bins.bin_edges,
        ).data
        for locdata in locdatas
    ]

    if rescale is None:
        norm = mcolors.Normalize(vmin=np.min(imgs), vmax=np.max(imgs))
    else:
        norm = rescale
    imgs = [adjust_contrast(img, rescale=norm) for img in imgs]

    new = np.zeros_like(imgs[0])
    rgb_stack = np.stack([new] * 3, axis=2)

    for i, img in enumerate(imgs):
        rgb_stack[:, :, i] = img

    rgb_stack = np.transpose(rgb_stack, axes=(1, 0, 2))
    viewer.add_image(
        rgb_stack,
        **dict({"name": f"LocData {locdata_id}", "rgb": True}, **kwargs),
    )
    return viewer
