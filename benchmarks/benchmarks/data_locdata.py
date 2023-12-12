"""
Benchmark functions to be used with Airspeed Velocity.
"""
import locan as lc
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=1)


class BenchmarkMetadata:
    """
    Benchmarks for data.metadata_pb2.Metadata objects
    """

    def setup(self):
        self.metadata = lc.data.metadata_pb2.Metadata()

    def time_initialize_metadata(self):
        lc.data.metadata_pb2.Metadata()

    def mem_metadata(self):
        return self.metadata


class BenchmarkLocData:
    """
    Benchmarks for initializing LocData objects
    """

    def setup(self):
        n_points = 1_000_000
        coordinates = rng.uniform(low=0, high=10_000, size=(4, n_points))
        locdata_dict = {
            "position_x": coordinates[0],
            "position_y": coordinates[1],
            "position_z": coordinates[2],
            "frame": np.linspace(0, n_points, n_points, dtype=int),
            "intensity": coordinates[3],
        }
        self.locdata_df = pd.DataFrame(locdata_dict)
        self.meta = lc.data.metadata_pb2.Metadata()
        self.meta.creation_time.seconds = 1
        self.locdata_3d = lc.LocData.from_dataframe(
            dataframe=self.locdata_df, meta=self.meta
        )

        n_selected_points = 1_000
        self.selection_indices = rng.integers(
            low=0, high=n_points, size=n_selected_points
        ).tolist()
        self.selection = lc.LocData.from_selection(
            locdata=self.locdata_3d, indices=self.selection_indices
        )

        self.locdatas = [
            lc.LocData.from_selection(locdata=self.locdata_3d, indices=idxs.tolist())
            for idxs in rng.integers(low=0, high=n_points, size=(10, n_selected_points))
        ]
        self.collection = lc.LocData.from_collection(locdatas=self.locdatas)

    def time_initialize_locdata_from_dataframe(self):
        lc.LocData.from_dataframe(dataframe=self.locdata_df, meta=self.meta)

    def time_locdata_selection(self):
        lc.LocData.from_selection(
            locdata=self.locdata_3d, indices=self.selection_indices
        )

    def time_locdata_collection(self):
        lc.LocData.from_collection(locdatas=self.locdatas)

    def mem_empty_locdata(self):
        return lc.LocData()

    def mem_locdata_from_dataframe(self):
        return self.locdata_3d

    def mem_locdata_selection(self):
        return self.selection

    def mem_locdata_selection_reduce(self):
        return self.selection.reduce()

    def mem_locdata_collection(self):
        return self.collection

    def mem_locdata_collection_reduce(self):
        return self.collection.reduce()


if __name__ == "__main__":
    class_ = BenchmarkLocData()
    class_.setup()
    print(class_.collection.data)
