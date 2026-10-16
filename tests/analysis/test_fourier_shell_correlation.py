import matplotlib.pyplot as plt  # this import is needed for visual inspection
import numpy as np
import pytest

import locan as lc
from locan import (
    FourierShellCorrelation,
    FourierShellCorrelationResults,
    Image,
    LocData,
    ranges,
)
from locan.analysis.fourier_shell_correlation import (
    compute_fourier_shell_correlation,
    preprocess_images,
)
from tests import TEST_DIR

# fixtures


@pytest.fixture()
def clathrin_3d():
    file_path = TEST_DIR / "test_data/clathrin_3d.asdf"
    locdata = lc.load_asdf_file(path=file_path)
    return locdata


# tests


class Test_preprocess_images:

    def test_preprocess_images(self, clathrin_3d):
        # get two images
        locdata = clathrin_3d
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )

        bin_range = ranges(locdata=locdata)
        bins = lc.Bins(bin_range=bin_range, bin_size=50)
        image_data, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )

        # test
        image, other_image = preprocess_images(image=image_data, other_image=image_data)

        assert image.shape == (79, 83, 17)
        assert other_image.shape == (79, 83, 17)

    def test_preprocess_images_differently_sized_images(self, clathrin_3d):
        # get two images
        locdata = clathrin_3d
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )

        bin_range = ranges(locdata=locdata)
        bins = lc.Bins(bin_range=bin_range, bin_size=50)
        image_data, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )

        bin_range = [(item[0], item[0] + (item[1] - item[0]) / 2) for item in bin_range]
        bins = lc.Bins(bin_range=bin_range, bin_size=50)
        other_image_data, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )

        # test
        image, other_image = preprocess_images(
            image=image_data, other_image=other_image_data
        )

        assert image.shape == (79, 83, 17)
        assert other_image.shape == (79, 83, 17)

    def test_preprocess_images_with_window(self, clathrin_3d):
        # get two images
        locdata = clathrin_3d
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )

        bin_range = ranges(locdata=locdata)
        bins = lc.Bins(bin_range=bin_range, bin_size=50)
        image_data, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )

        bin_range = [(item[0], item[0] + (item[1] - item[0]) / 2) for item in bin_range]
        bins = lc.Bins(bin_range=bin_range, bin_size=50)
        other_image_data, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )

        # test
        image, other_image = preprocess_images(
            image=image_data, other_image=other_image_data, window=("tukey", 0.75)
        )

        assert image.shape == (79, 83, 17)
        assert other_image.shape == (79, 83, 17)

        @pytest.mark.visual
        def test_preprocess_images(self, clathrin_3d):
            # get two images
            locdata = clathrin_3d
            locdata_chunks = LocData.from_chunks(
                locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
            )

            bin_range = ranges(locdata=locdata)
            bins = lc.Bins(bin_range=bin_range, bin_size=50)
            image_data, bins, labels = lc.histogram(
                locdata=locdata_chunks.references[0], bins=bins
            )

            bin_range = [
                (item[0], item[0] + (item[1] - item[0]) / 2) for item in bin_range
            ]
            bins = lc.Bins(bin_range=bin_range, bin_size=50)
            other_image_data, other_bins, other_labels = lc.histogram(
                locdata=locdata_chunks.references[1], bins=bins
            )

            plt.imshow(image_data)
            plt.show()

            plt.imshow(other_image_data)
            plt.show()

            # test
            image, other_image = preprocess_images(
                image=image_data, other_image=other_image_data
            )

            print(image.shape, other_image.shape)
            assert image.shape == (79, 83, 17)
            assert other_image.shape == (79, 83, 17)

            plt.imshow(image)
            plt.show()

            plt.imshow(other_image)
            plt.show()

            image, other_image = preprocess_images(
                image=image_data, other_image=other_image_data, window=("tukey", 0.75)
            )

            print(image.shape, other_image.shape)
            assert image.shape == (79, 83, 17)
            assert other_image.shape == (79, 83, 17)

            plt.imshow(image)
            plt.show()

            plt.imshow(other_image)
            plt.show()

            plt.close("all")


class Test_compute_fourier_shell_correlation:

    def test_compute_fourier_shell_correlation(self, clathrin_3d):
        # get two images
        locdata = clathrin_3d
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )
        bins = lc.Bins(
            bin_range=np.vstack(
                (
                    locdata.coordinates.min(axis=0),
                    locdata.coordinates.max(axis=0),
                )
            ).T,
            bin_size=50,
        )
        image, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )
        other_image, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )

        # test
        result = compute_fourier_shell_correlation(
            image=image, other_image=other_image, bin_size=50
        )
        assert result.frequency.shape == (7,)
        assert result.correlation.shape == (7,)

    @pytest.mark.visual
    def test_compute_fourier_shell_correlation_visual_(self, clathrin_3d):
        # get two images
        locdata = clathrin_3d
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )
        bins = lc.Bins(
            bin_range=np.vstack(
                (
                    locdata.coordinates.min(axis=0),
                    locdata.coordinates.max(axis=0),
                )
            ).T,
            bin_size=50,
        )
        image, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )
        other_image, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )
        print(bins, labels)

        # test
        result = compute_fourier_shell_correlation(
            image=image, other_image=other_image, bin_size=50
        )
        assert result.frequency.shape == (7,)
        assert result.correlation.shape == (7,)

        fsc_smooth = result.correlation
        # fsc_smooth = gaussian_filter1d(result.fsc, sigma=1)

        # plot
        plt.figure(figsize=(4, 3))
        plt.plot(result.frequency, fsc_smooth)
        plt.axhline(0.143, color="red", linestyle="--")
        plt.xlabel("Spatial frequency (µm⁻¹)")
        plt.ylabel("Fourier Shell Correlation")
        plt.grid(True)
        plt.show()


class TestFourierShellCorrelationResults:

    def test_init(self):
        fsc_results = FourierShellCorrelationResults(
            frequency=np.arange(5, dtype=np.float64),
            correlation=np.arange(5, 0, -1, dtype=np.float64),
            image=Image(),
            other_image=Image(),
        )
        assert repr(fsc_results) == "FourierShellCorrelationResults()"
        assert fsc_results.frequency.shape == (5,)
        assert fsc_results.correlation.shape == (5,)
        assert fsc_results.image
        assert fsc_results.other_image

        assert np.isnan(fsc_results.threshold_frequency())
        assert np.isnan(fsc_results.resolution())


class TestFourierShellCorrelation:

    def test_init(self):
        fsc = FourierShellCorrelation(
            bin_size=50,
        )
        assert (
            repr(fsc)
            == "FourierShellCorrelation(bin_size=50, smooth=False, window=None)"
        )
        assert fsc.results is None

    def test_empty_locdata(self, caplog):
        fsc = FourierShellCorrelation(
            bin_size=50,
        )
        fsc.compute(locdata=LocData())
        assert fsc.results is None

        fsc.plot()
        assert caplog.record_tuples == [
            ("locan.analysis.fourier_shell_correlation", 30, "Locdata is empty.")
        ]
        plt.close("all")

    def test_compute(self, clathrin_3d):
        fsc = FourierShellCorrelation(
            bin_size=50,
        ).compute(locdata=clathrin_3d)
        assert fsc.results.frequency.shape == (7,)
        assert fsc.results.correlation.shape == (7,)
        assert fsc.results.image.data.shape == (79, 83, 17)
        assert fsc.results.other_image.data.shape == (79, 83, 17)

    def test_compute_with_other_data(self, clathrin_3d):
        locdata_chunks = LocData.from_chunks(
            locdata=clathrin_3d, n_chunks=2, order="alternating"
        )

        fsc = FourierShellCorrelation(
            bin_size=50,
        ).compute(
            locdata=locdata_chunks.references[0],
            other_locdata=locdata_chunks.references[1],
        )
        assert fsc.results.frequency.shape == (7,)
        assert fsc.results.correlation.shape == (7,)
        assert fsc.results.image.data.shape == (79, 83, 17)
        assert fsc.results.other_image.data.shape == (79, 83, 17)

    def test_smoothing(self, clathrin_3d):
        fsc = FourierShellCorrelation(
            bin_size=50,
            smooth=1,
        ).compute(locdata=clathrin_3d)
        assert fsc.results.frequency.shape == (7,)
        assert fsc.results.correlation.shape == (7,)
        assert fsc.results.image.data.shape == (79, 83, 17)
        assert fsc.results.other_image.data.shape == (79, 83, 17)

    def test_window(self, clathrin_3d):
        fsc = FourierShellCorrelation(
            bin_size=50,
            window=("tukey", 0.8),
        ).compute(locdata=clathrin_3d)
        print(fsc.results)
        assert fsc.results.frequency.shape == (7,)
        assert fsc.results.correlation.shape == (7,)
        assert fsc.results.image.data.shape == (79, 83, 17)
        assert fsc.results.other_image.data.shape == (79, 83, 17)

    @pytest.mark.visual
    def test_plot(self, clathrin_3d):
        fsc = FourierShellCorrelation(
            bin_size=10,
            smooth=1,
            window=("tukey", 0.8),
        ).compute(locdata=clathrin_3d)

        fsc.plot()
        plt.show()

        plt.close("all")
