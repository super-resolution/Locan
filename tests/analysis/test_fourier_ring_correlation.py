import matplotlib.pyplot as plt  # this import is needed for visual inspection
import numpy as np
import pytest

import locan as lc
from locan import (
    FourierRingCorrelation,
    FourierRingCorrelationResults,
    Image,
    LocData,
    ranges,
)
from locan.analysis.fourier_ring_correlation import (
    compute_fourier_ring_correlation,
    get_frequency_at_threshold,
    preprocess_images,
)
from tests import TEST_DIR

# fixtures


@pytest.fixture()
def npc_gp210():
    file_path = TEST_DIR / "test_data/npc_gp210.asdf"
    locdata = lc.load_asdf_file(path=file_path)
    return locdata


# tests


class Test_preprocess_images:

    def test_preprocess_images(self, npc_gp210):
        # get two images
        locdata = npc_gp210
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )

        bin_range = ranges(locdata=locdata)
        bins = lc.Bins.from_bin_size(10, bin_range)
        image_data, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )

        # test
        image, other_image = preprocess_images(image=image_data, other_image=image_data)

        assert image.shape == (20, 23)
        assert other_image.shape == (20, 23)

    def test_preprocess_images_differently_sized_images(self, npc_gp210):
        # get two images
        locdata = npc_gp210
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )

        bin_range = ranges(locdata=locdata)
        bins = lc.Bins.from_bin_size(10, bin_range)
        image_data, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )

        bin_range = [(item[0], item[0] + (item[1] - item[0]) / 2) for item in bin_range]
        bins = lc.Bins.from_bin_size(10, bin_range)
        other_image_data, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )

        # test
        image, other_image = preprocess_images(
            image=image_data, other_image=other_image_data
        )

        assert image.shape == (20, 23)
        assert other_image.shape == (20, 23)

    def test_preprocess_images_with_window(self, npc_gp210):
        # get two images
        locdata = npc_gp210
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )

        bin_range = ranges(locdata=locdata)
        bins = lc.Bins.from_bin_size(10, bin_range)
        image_data, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )

        bin_range = [(item[0], item[0] + (item[1] - item[0]) / 2) for item in bin_range]
        bins = lc.Bins.from_bin_size(10, bin_range)
        other_image_data, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )

        # test
        image, other_image = preprocess_images(
            image=image_data, other_image=other_image_data, window=("tukey", 0.75)
        )

        assert image.shape == (20, 23)
        assert other_image.shape == (20, 23)

        @pytest.mark.visual
        def test_preprocess_images(self, npc_gp210):
            # get two images
            locdata = npc_gp210
            locdata_chunks = LocData.from_chunks(
                locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
            )

            bin_range = ranges(locdata=locdata)
            bins = lc.Bins.from_bin_size(10, bin_range)
            image_data, bins, labels = lc.histogram(
                locdata=locdata_chunks.references[0], bins=bins
            )

            bin_range = [
                (item[0], item[0] + (item[1] - item[0]) / 2) for item in bin_range
            ]
            bins = lc.Bins.from_bin_size(10, bin_range)
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
            assert image.shape == (20, 23)
            assert other_image.shape == (20, 23)

            plt.imshow(image)
            plt.show()

            plt.imshow(other_image)
            plt.show()

            image, other_image = preprocess_images(
                image=image_data, other_image=other_image_data, window=("tukey", 0.75)
            )

            print(image.shape, other_image.shape)
            assert image.shape == (20, 23)
            assert other_image.shape == (20, 23)

            plt.imshow(image)
            plt.show()

            plt.imshow(other_image)
            plt.show()

            plt.close("all")


class Test_compute_fourier_ring_correlation:

    def test_compute_fourier_ring_correlation(self, npc_gp210):
        # get two images
        locdata = npc_gp210
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )
        bins = lc.Bins.from_bin_size(
            bin_range=np.vstack(
                (
                    locdata.coordinates.min(axis=0),
                    locdata.coordinates.max(axis=0),
                )
            ).T,
            bin_size=10,
        )
        image, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )
        other_image, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )

        # test
        result = compute_fourier_ring_correlation(
            image=image, other_image=other_image, bin_size=10
        )
        assert result.frequency.shape == (9,)
        assert result.correlation.shape == (9,)

    @pytest.mark.visual
    def test_compute_fourier_ring_correlation_visual_(self, npc_gp210):
        # get two images
        locdata = npc_gp210
        locdata_chunks = LocData.from_chunks(
            locdata=locdata, chunk_size=len(locdata) // 2, order="alternating"
        )
        bins = lc.Bins.from_bin_size(
            bin_range=np.vstack(
                (
                    locdata.coordinates.min(axis=0),
                    locdata.coordinates.max(axis=0),
                )
            ).T,
            bin_size=10,
        )
        image, bins, labels = lc.histogram(
            locdata=locdata_chunks.references[0], bins=bins
        )
        other_image, other_bins, other_labels = lc.histogram(
            locdata=locdata_chunks.references[1], bins=bins
        )
        # print(bins, labels, bins.bin_range)

        # test
        result = compute_fourier_ring_correlation(
            image=image, other_image=other_image, bin_size=10
        )
        assert result.frequency.shape == (9,)
        assert result.correlation.shape == (9,)

        frc_smooth = result.correlation
        # frc_smooth = gaussian_filter1d(result.frc, sigma=1)

        # plot
        plt.figure(figsize=(4, 3))
        plt.plot(result.frequency, frc_smooth)
        plt.axhline(0.143, color="red", linestyle="--")
        plt.xlabel("Spatial frequency (µm⁻¹)")
        plt.ylabel("Fourier Ring Correlation")
        plt.grid(True)
        plt.show()


def test_get_frequency_at_threshold(npc_gp210):
    frequencies = np.arange(5, dtype=np.float64)
    correlation = np.arange(1, 0, -0.2, dtype=np.float64)

    freq_cross = get_frequency_at_threshold(
        frequencies=frequencies, correlation=correlation, threshold=0.52
    )
    assert freq_cross == pytest.approx(2.4, rel=0.1)

    plt.plot(frequencies, correlation, marker="o")
    plt.axhline(0.52, color="red", linestyle="--")
    plt.axvline(freq_cross, color="red", linestyle="--")
    # plt.show()

    plt.close("all")


class TestFourierRingCorrelationResults:

    def test_init(self):
        frc_results = FourierRingCorrelationResults(
            frequency=np.arange(5, dtype=np.float64),
            correlation=np.arange(5, 0, -1, dtype=np.float64),
            image=Image(),
            other_image=Image(),
        )
        assert repr(frc_results) == "FourierRingCorrelationResults()"
        assert frc_results.frequency.shape == (5,)
        assert frc_results.correlation.shape == (5,)
        assert frc_results.image
        assert frc_results.other_image

        assert np.isnan(frc_results.threshold_frequency())
        assert np.isnan(frc_results.resolution())


class TestFourierRingCorrelation:

    def test_init(self):
        frc = FourierRingCorrelation(
            bin_size=10,
        )
        assert (
            repr(frc)
            == "FourierRingCorrelation(bin_size=10, smooth=False, window=None)"
        )
        assert frc.results is None

    def test_empty_locdata(self, caplog):
        frc = FourierRingCorrelation(
            bin_size=10,
        )
        frc.compute(locdata=LocData())
        assert frc.results is None

        frc.plot()
        assert caplog.record_tuples == [
            ("locan.analysis.fourier_ring_correlation", 30, "Locdata is empty.")
        ]
        plt.close("all")

    def test_compute(self, npc_gp210):
        frc = FourierRingCorrelation(
            bin_size=10,
        ).compute(locdata=npc_gp210)
        assert frc.results.frequency.shape == (9,)
        assert frc.results.correlation.shape == (9,)
        assert frc.results.image.data.shape == (20, 23)
        assert frc.results.other_image.data.shape == (20, 23)

    def test_compute_with_other_data(self, npc_gp210):
        locdata_chunks = LocData.from_chunks(
            locdata=npc_gp210, n_chunks=2, order="successive"
        )

        frc = FourierRingCorrelation(
            bin_size=10,
        ).compute(
            locdata=locdata_chunks.references[0],
            other_locdata=locdata_chunks.references[1],
        )
        assert frc.results.frequency.shape == (9,)
        assert frc.results.correlation.shape == (9,)
        assert frc.results.image.data.shape == (20, 23)
        assert frc.results.other_image.data.shape == (20, 23)

    def test_smoothing(self, npc_gp210):
        frc = FourierRingCorrelation(
            bin_size=10,
            smooth=1,
        ).compute(locdata=npc_gp210)
        assert frc.results.frequency.shape == (9,)
        assert frc.results.correlation.shape == (9,)
        assert frc.results.image.data.shape == (20, 23)
        assert frc.results.other_image.data.shape == (20, 23)

    def test_window(self, npc_gp210):
        frc = FourierRingCorrelation(
            bin_size=10,
            window=("tukey", 0.8),
        ).compute(locdata=npc_gp210)
        print(frc.results)
        assert frc.results.frequency.shape == (9,)
        assert frc.results.correlation.shape == (9,)
        assert frc.results.image.data.shape == (20, 23)
        assert frc.results.other_image.data.shape == (20, 23)

    @pytest.mark.visual
    def test_plot(self, npc_gp210):
        frc = FourierRingCorrelation(
            bin_size=10,
            smooth=1,
            window=("tukey", 0.8),
        ).compute(locdata=npc_gp210)

        plt.imshow(frc.results.image.data)
        plt.show()

        plt.imshow(frc.results.other_image.data)
        plt.show()

        frc.plot()
        plt.show()

        plt.close("all")
