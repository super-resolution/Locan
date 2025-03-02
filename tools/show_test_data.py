"""
Utility to visualize test data.

Show all test datasets that are available in /tests/test_data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif

import locan as lc

TEST_DIR: Path = Path(__file__).parents[1] / "tests"
PATH_TEST_DATA = TEST_DIR / "test_data"


def get_files(path: Path = PATH_TEST_DATA) -> list[Path]:
    files = path.rglob(pattern="*.*")
    file_list = [file.relative_to(path) for file in files]
    return file_list


def show_decode_dstorm_data() -> None:
    file = "decode_dstorm_data.h5"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_decode_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.loc[:, ["position_x", "position_y", "position_z"]].describe())
    lc.render_2d_mpl(
        locdata=locdata.projection(coordinate_labels=["position_x", "position_y"]),
        bin_size=0.1,
    )
    lc.scatter_2d_mpl(
        locdata=locdata.projection(coordinate_labels=["position_x", "position_y"]),
        index=False,
    )
    plt.show()


def show_decode_dstorm_data_empty() -> None:
    file = "decode_dstorm_data_empty.h5"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_decode_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    lc.render_2d_mpl(
        locdata=locdata.projection(coordinate_labels=["position_x", "position_y"]),
        bin_size=0.1,
    )
    lc.scatter_2d_mpl(
        locdata=locdata.projection(coordinate_labels=["position_x", "position_y"]),
        index=False,
    )
    plt.show()


def show_Elyra_dstorm_data() -> None:
    file = "Elyra_dstorm_data.txt"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_Elyra_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.loc[:, ["position_x", "position_y"]].describe())
    lc.render_2d_mpl(locdata=locdata, bin_size=100)
    plt.show()


def show_five_blobs() -> None:
    file = "five_blobs.txt"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_txt_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.loc[:, ["position_x", "position_y"]].describe())
    lc.render_2d_mpl(locdata=locdata, bin_size=10)
    lc.scatter_2d_mpl(locdata=locdata, index=False)
    plt.show()


def show_five_blobs_3D() -> None:
    import napari

    file = "five_blobs_3D.txt"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_txt_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.describe())
    viewer = lc.render_3d_napari(
        locdata=locdata,
        bin_size=10,
        cmap="viridis",
        gamma=0.1,
    )
    viewer.add_points(locdata.coordinates)
    napari.run()


def show_images() -> None:
    import napari

    file = "images.tif"
    file_path = PATH_TEST_DATA / file
    image_stack = tif.imread(str(file_path))

    print("\n" + file + "\n")
    print("shape:", np.shape(image_stack))  # type: ignore

    viewer = napari.Viewer()
    viewer.add_image(image_stack, name="images")
    napari.run()


def show_Nanoimager_dstorm_data() -> None:
    file = "Nanoimager_dstorm_data.csv"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_Nanoimager_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.describe())
    lc.render_2d_mpl(
        locdata=locdata.projection(coordinate_labels=["position_x", "position_y"]),
        bin_size=100,
        rescale=lc.Trafo.EQUALIZE,
    )
    plt.show()


def show_npc_gp210_data() -> None:
    file = "npc_gp210.asdf"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_asdf_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.describe())
    lc.render_2d_mpl(
        locdata=locdata.projection(coordinate_labels=["position_x", "position_y"]),
        bin_size=10,
        rescale=lc.Trafo.EQUALIZE,
    )
    plt.show()


def show_protobuf_message_metadata_pb2_Metadata_v0p11() -> None:
    file = "protobuf_message_metadata_pb2.Metadata_v0p11"
    file_path = PATH_TEST_DATA / file
    metadata_new = lc.data.metadata_pb2.Metadata()
    with open(file_path, "rb") as file_:
        metadata_new.ParseFromString(file_.read())
    print("\n" + file + "\n")
    print(metadata_new)


def show_protobuf_message_metadata_pb2_Metadata_v0p12() -> None:
    file = "protobuf_message_metadata_pb2.Metadata_v0p12"
    file_path = PATH_TEST_DATA / file
    metadata_new = lc.data.metadata_pb2.Metadata()
    with open(file_path, "rb") as file_:
        metadata_new.ParseFromString(file_.read())
    print("\n" + file + "\n")
    print(metadata_new)


def show_rapidSTORM_dstorm_data() -> None:
    file = "rapidSTORM_dstorm_data.txt"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_rapidSTORM_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.describe())
    lc.render_2d_mpl(
        locdata=locdata,
        bin_size=100,
        rescale=lc.Trafo.EQUALIZE,
    )
    plt.show()


def show_rapidSTORM_dstorm_track_data() -> None:
    file = "rapidSTORM_dstorm_track_data.txt"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_rapidSTORM_track_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.describe())
    lc.render_2d_mpl(
        locdata=locdata,
        bin_size=100,
    )
    lc.scatter_2d_mpl(locdata=locdata, index=False)
    plt.show()


def show_rapidSTORM_from_images() -> None:
    file = "rapidSTORM_from_images.txt"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_rapidSTORM_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.describe())
    lc.render_2d_mpl(
        locdata=locdata,
        bin_size=100,
        rescale=lc.Trafo.EQUALIZE,
    )
    lc.scatter_2d_mpl(locdata=locdata, index=False)
    plt.show()


def show_smap_dstorm_data() -> None:
    import napari

    file = "smap_dstorm_data.mat"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_SMAP_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.loc[:, ["position_x", "position_y", "position_z"]].describe())
    lc.render_3d_napari(
        locdata=locdata,
        bin_size=100,
        cmap="viridis",
    )
    napari.run()


def show_SMLM_dstorm_data() -> None:
    file = "SMLM_dstorm_data.smlm"
    file_path = PATH_TEST_DATA / file
    locdata: lc.LocData = lc.load_SMLM_file(path=file_path)  # type: ignore
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.describe())
    lc.render_2d_mpl(
        locdata=locdata,
        bin_size=100,
    )
    lc.scatter_2d_mpl(locdata=locdata, index=False)
    plt.show()


def show_Thunderstorm_dstorm_data() -> None:
    file = "Thunderstorm_dstorm_data.csv"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_thunderstorm_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.loc[:, ["position_x", "position_y"]].describe())
    lc.render_2d_mpl(
        locdata=locdata,
        bin_size=100,
        rescale=lc.Trafo.EQUALIZE,
    )
    plt.show()


def show_transform_BunwarpJ_transformation_elastic_green() -> None:
    file = "transform/BunwarpJ_transformation_elastic_green.txt"
    file_path = PATH_TEST_DATA / file
    with open(file_path, "rb") as file_:
        data = file_.read()
    print("\n" + file + "\n")
    print(data)


def show_transform_BunwarpJ_transformation_raw_green() -> None:
    file = "transform/BunwarpJ_transformation_raw_green.txt"
    file_path = PATH_TEST_DATA / file
    with open(file_path, "rb") as file_:
        data = file_.read()
    print("\n" + file + "\n")
    print(data)


def show_transform_rapidSTORM_beads_green() -> None:
    file = "transform/rapidSTORM_beads_green.asdf"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_asdf_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.loc[:, ["position_x", "position_y"]].describe())
    lc.render_2d_mpl(
        locdata=locdata,
        bin_size=1,
        rescale=lc.Trafo.EQUALIZE,
    )
    plt.show()


def show_transform_rapidSTORM_beads_red() -> None:
    file = "transform/rapidSTORM_beads_red.asdf"
    file_path = PATH_TEST_DATA / file
    locdata = lc.load_asdf_file(path=file_path)
    print("\n" + file + "\n")
    print(locdata.meta)
    print(list(locdata.data.columns), "\n")
    print(locdata.data.loc[:, ["position_x", "position_y"]].describe())
    lc.render_2d_mpl(
        locdata=locdata,
        bin_size=1,
        rescale=lc.Trafo.EQUALIZE,
    )
    plt.show()


if __name__ == "__main__":
    for f in get_files():
        print(f)
    # show_decode_dstorm_data()
    # show_decode_dstorm_data_empty()
    # show_Elyra_dstorm_data()
    # show_five_blobs()
    # show_five_blobs_3D()
    # show_images()
    # show_Nanoimager_dstorm_data()
    # show_npc_gp210_data()
    # show_protobuf_message_metadata_pb2_Metadata_v0p11()
    # show_protobuf_message_metadata_pb2_Metadata_v0p12()
    # show_rapidSTORM_dstorm_data()
    # show_rapidSTORM_dstorm_track_data()
    # show_rapidSTORM_from_images()
    # show_smap_dstorm_data()
    # show_SMLM_dstorm_data()
    # show_Thunderstorm_dstorm_data()
    # show_transform_BunwarpJ_transformation_elastic_green()
    # show_transform_BunwarpJ_transformation_raw_green()
    # show_transform_rapidSTORM_beads_green()
    # show_transform_rapidSTORM_beads_red()
