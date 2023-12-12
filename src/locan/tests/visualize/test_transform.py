import numpy as np
import pytest

from locan import HistogramEqualization, Trafo, adjust_contrast
from locan.visualize.transform import Transform


def test_Trafo():
    assert list(Trafo)


def test_HistogramEqualization():
    img = np.array((1, 2, 3, 9), dtype=np.float64)
    norm = HistogramEqualization(power=1, n_bins=4)
    print(norm)
    assert isinstance(norm, Transform)
    new_img = norm(img)
    assert np.allclose(new_img, [0.5, 0.5, 0.625, 1.0])

    norm = HistogramEqualization(power=1, n_bins=4, vmax=3)
    assert isinstance(norm, Transform)
    new_img = norm(img)
    assert np.allclose(new_img, [0.25, 0.375, 1.0, 1.0])

    norm = HistogramEqualization(power=1, n_bins=4, vmin=3)
    assert isinstance(norm, Transform)
    new_img = norm(img)
    assert np.allclose(new_img, [0.75, 0.75, 0.75, 1.0])

    norm = HistogramEqualization(power=1, n_bins=3, mask=img > 1)
    assert isinstance(norm, Transform)
    new_img = norm(img)
    assert np.allclose(new_img, [0.66666667, 0.66666667, 0.66666667, 1.0])

    norm = HistogramEqualization(power=3, n_bins=3)
    new_img = norm(img)
    assert np.allclose(new_img, [0.96428571, 0.96428571, 0.96428571, 1.0])

    norm = HistogramEqualization(power=0.3, n_bins=3)
    new_img = norm(img)
    assert np.allclose(new_img, [0.58165808, 0.58165808, 0.58165808, 1.0])


def test_adjust_contrast():
    img = np.array((1, 2, 3, 9), dtype=np.float64)
    new_img = adjust_contrast(img, rescale=(50, 100))
    assert np.array_equal(new_img, np.array((0, 0, 0, 1)))

    img = np.array((1, 2, 3, 9), dtype=np.uint8)
    new_img = adjust_contrast(img, rescale=None)
    assert np.array_equal(img, new_img)

    new_img = adjust_contrast(img, rescale=False)
    assert np.array_equal(img, new_img)

    new_img = adjust_contrast(img, rescale="equal")
    assert max(new_img) == 1

    new_img = adjust_contrast(img, rescale=True)
    # assert np.array_equal(new_img, np.array((0, 31, 63, 255)))
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.0)))

    new_img = adjust_contrast(img, rescale=(0, 50))
    assert np.array_equal(new_img, np.array((0, 63, 127, 255)))

    new_img = adjust_contrast(img, out_range=(0, 10))
    assert np.array_equal(new_img, np.array((0, 1.25, 2.5, 10)))

    new_img = adjust_contrast(img * 1.0, rescale=True)
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.0)))

    new_img = adjust_contrast(img, rescale="unity")
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.0)))

    norm = HistogramEqualization(power=1, n_bins=4)
    new_img = adjust_contrast(img, rescale=norm)
    assert np.allclose(new_img, [0.5, 0.5, 0.625, 1.0])


def test_transform():
    img = np.array((2, 2.5, 3, 9), dtype=np.float64)
    img_with_zeros = np.array((2, 0, 2.5, 3, 9), dtype=np.float64)

    # 0 No transformation
    new_img = adjust_contrast(img, rescale=0)
    assert new_img is img
    new_img = adjust_contrast(img, rescale=None)
    assert new_img is img
    new_img = adjust_contrast(img, rescale=False)
    assert new_img is img
    new_img = adjust_contrast(img, rescale=Trafo.NONE)
    assert new_img is img

    # 1) standardize: rescale (min, max) to (0, 1)
    new_img = adjust_contrast(img, rescale=1)
    assert new_img.dtype == float
    assert (new_img.min(), new_img.max()) == (0, 1)

    new_img = adjust_contrast(img, rescale=Trafo.STANDARDIZE)
    assert new_img.dtype == float
    assert (new_img.min(), new_img.max()) == (0, 1)

    new_img = adjust_contrast(img, rescale="standardize")
    assert new_img.dtype == float
    assert (new_img.min(), new_img.max()) == (0, 1)

    # 2) standardize_uint8: rescale (min, max) to (0, 255)
    new_img = adjust_contrast(img, rescale=Trafo.STANDARDIZE_UINT8)
    assert new_img.dtype == np.uint8
    assert (new_img.min(), new_img.max()) == (0, 255)

    # 3) zero: (0, max) to (0, 1)
    new_img = adjust_contrast(img, rescale=Trafo.ZERO)
    assert (new_img.min(), new_img.max()) == (pytest.approx(0.2222222222222222), 1)

    # 4) zero_uint8: (0, max) to (0, 255)
    new_img = adjust_contrast(img, rescale=Trafo.ZERO_UINT8)
    assert new_img.dtype == np.uint8
    assert (new_img.min(), new_img.max()) == (56, 255)

    # 5) equalize: equalize histogram for all values>0
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE)
    assert new_img.dtype == np.float64
    assert np.allclose(new_img, [0.25, 0.0, 0.48611111, 0.70833333, 1.0])

    # 6) equalize_uint8: equalize histogram for all values>0 within (0, 255)
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_UINT8)
    assert new_img.dtype == np.uint8
    assert np.allclose(new_img, [63, 0, 123, 180, 255])

    # 7) equalize_all: equalize histogram
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_ALL)
    assert new_img.dtype == np.float64
    assert np.allclose(new_img, [0.4, 0.2, 0.58888889, 0.76666667, 1.0])

    # 8) equalize_all_uint8: equalize histogram within (0, 255)
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_ALL_UINT8)
    assert new_img.dtype == np.uint8
    assert np.allclose(new_img, [102, 51, 150, 195, 255])

    # 9) equalize_0p3: equalize histogram with factor=0.3 for all values>0
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3)
    assert new_img.dtype == np.float64
    assert np.allclose(new_img, [0.25, 0.0, 0.48611111, 0.70833333, 1.0])
    # result is the same as for EQUALIZE due to the large number of bins

    # 10) equalize_0p3_uint8: equalize histogram with factor=0.3 for all values>0 within (0, 255)
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3_UINT8)
    assert new_img.dtype == np.uint8
    assert np.allclose(new_img, [63, 0, 123, 180, 255])

    # 11) equalize_0p3_all: equalize histogram with factor=0.3
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3_ALL)
    assert new_img.dtype == np.float64
    assert np.allclose(new_img, [0.4, 0.2, 0.58888889, 0.76666667, 1.0])

    # 12) equalize_0p3_all_uint8: equalize histogram with factor=0.3 within (0, 255)
    new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3_ALL_UINT8)
    assert new_img.dtype == np.uint8
    assert np.allclose(new_img, [102, 51, 150, 195, 255])


def test_transform_with_nan():
    img = np.array((2, np.nan, 3, 9), dtype=np.float64)
    img_with_zeros = np.array((2, 0, np.nan, 3, 9), dtype=np.float64)

    # 0 No transformation
    new_img = adjust_contrast(img, rescale=0)
    assert new_img is img
    new_img = adjust_contrast(img, rescale=None)
    assert new_img is img
    new_img = adjust_contrast(img, rescale=False)
    assert new_img is img
    new_img = adjust_contrast(img, rescale=Trafo.NONE)
    assert new_img is img

    # 1) standardize: rescale (min, max) to (0, 1)
    new_img = adjust_contrast(img, rescale=1)
    assert new_img.dtype == float
    assert (np.nanmin(new_img), np.nanmax(new_img)) == (0, 1)

    new_img = adjust_contrast(img, rescale=Trafo.STANDARDIZE)
    assert new_img.dtype == float
    assert (np.nanmin(new_img), np.nanmax(new_img)) == (0, 1)

    new_img = adjust_contrast(img, rescale="standardize")
    assert new_img.dtype == float
    assert (np.nanmin(new_img), np.nanmax(new_img)) == (0, 1)

    # 2) standardize_uint8: rescale (min, max) to (0, 255)
    with pytest.warns(RuntimeWarning):
        new_img = adjust_contrast(img, rescale=Trafo.STANDARDIZE_UINT8)
    assert new_img.dtype == np.uint8
    assert (np.nanmin(new_img), np.nanmax(new_img)) == (0, 255)

    # 3) zero: (0, max) to (0, 1)
    new_img = adjust_contrast(img, rescale=Trafo.ZERO)
    assert (np.nanmin(new_img), np.nanmax(new_img)) == (
        pytest.approx(0.2222222222222222),
        1,
    )

    # 4) zero_uint8: (0, max) to (0, 255)
    with pytest.warns(RuntimeWarning):
        new_img = adjust_contrast(img, rescale=Trafo.ZERO_UINT8)
    assert new_img.dtype == np.uint8
    assert (np.nanmin(new_img), np.nanmax(new_img)) == (0, 255)
    # note: nan is converted to 0

    # 5) equalize: equalize histogram for all values>0
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE)
        assert new_img.dtype == np.float64
        assert np.allclose(new_img, [0.25, 0.0, 0.48611111, 0.70833333, 1.0])

    # 6) equalize_uint8: equalize histogram for all values>0 within (0, 255)
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_UINT8)
        assert new_img.dtype == np.uint8
        assert np.allclose(new_img, [63, 0, 123, 180, 255])

    # 7) equalize_all: equalize histogram
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_ALL)
        assert new_img.dtype == np.float64
        assert np.allclose(new_img, [0.4, 0.2, 0.58888889, 0.76666667, 1.0])

    # 8) equalize_all_uint8: equalize histogram within (0, 255)
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_ALL_UINT8)
        assert new_img.dtype == np.uint8
        assert np.allclose(new_img, [102, 51, 150, 195, 255])

    # 9) equalize_0p3: equalize histogram with factor=0.3 for all values>0
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3)
        assert new_img.dtype == np.float64
        assert np.allclose(new_img, [0.25, 0.0, 0.48611111, 0.70833333, 1.0])
        # result is the same as for EQUALIZE due to the large number of bins

    # 10) equalize_0p3_uint8: equalize histogram with factor=0.3
    # for all values>0 within (0, 255)
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3_UINT8)
        assert new_img.dtype == np.uint8
        assert np.allclose(new_img, [63, 0, 123, 180, 255])

    # 11) equalize_0p3_all: equalize histogram with factor=0.3
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3_ALL)
        assert new_img.dtype == np.float64
        assert np.allclose(new_img, [0.4, 0.2, 0.58888889, 0.76666667, 1.0])

    # 12) equalize_0p3_all_uint8: equalize histogram with factor=0.3 within (0, 255)
    with pytest.raises(ValueError):
        new_img = adjust_contrast(img_with_zeros, rescale=Trafo.EQUALIZE_0P3_ALL_UINT8)
        assert new_img.dtype == np.uint8
        assert np.allclose(new_img, [102, 51, 150, 195, 255])
