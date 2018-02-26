"""
This module provides a template for a specialized analysis class.
It also provides helper functions to be used in specialized analysis classes.
And it provides standard interface functions to be used in specialized analysis classes.
"""
import time
from surepy.analysis import metadata_analysis_pb2



# Dealing with metadata

def _init_meta(self):
    meta_ = metadata_analysis_pb2.AMetadata()
    meta_.identifier = str(self.__class__.count)
    meta_.creation_date = int(time.time())
    meta_.method.name = str(self.__class__.__name__)
    meta_.method.parameter = str(self.parameter)
    return meta_


def _update_meta(self, meta=None):
    meta_ = self.meta
    if meta is None:
        pass
    else:
        try:
            meta_.MergeFrom(meta)
        except TypeError:
            for key, value in meta.items():
                setattr(meta_, key, value)

    return meta_

# saving data

def save_results(self, path):
    self.results.to_csv(path=path)