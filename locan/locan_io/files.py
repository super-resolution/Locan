"""
File manager

Identify, match, and group files to be batch-processed.
The class Files is a wrapper for a pandas.DataFrame with selected methods to
identify, match, and group file paths.
"""
from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path

try:
    from typing import Any, Iterable, Self
except ImportError:
    from typing import Any, Iterable, TypeVar

import pandas as pd

from locan.locan_io.utilities import find_file_upstream

__all__ = ["Files"]

logger = logging.getLogger(__name__)

try:
    Self
except NameError:
    Self = TypeVar("Self", bound="Files")


class Files:
    """
    Wrapper for a pandas.DataFrame with selected methods to
    identify, match, and group file paths.

    Notes:
    ------
    Iteration and indexing is implemented in a way that integer indexing
    or iterating over the Files instance returns
    a single row (as Series or namedtuple).
    Slice indexing returns a new Files instance with selected rows.

    Parameters
    ----------
    df : pd.DataFrame | dict[str, str] | None
        file names
    directory: str | bytes | os.PathLike | None
        base directory
    exists : bool
        raise `FileExistsError` if file in `df` does not exist
    column : str
        key/column in `df` from which to take a file list

    Attributes
    ----------
    df : pd.DataFrame
        dataframe carrying file paths
    directory: Path
        base directory
    """

    def __init__(
        self,
        df=None,
        directory=None,
        exists=True,
        column="file_path",
    ) -> None:
        if directory is None:
            self.directory = None
        elif isinstance(directory, (str, bytes, os.PathLike)):
            self.directory = Path(directory)
            if not self.directory.exists():
                raise ValueError(f"The directory {directory} does not exist.")
        else:
            raise TypeError("The given directory is not a valid type.")

        if df is None:
            self.df = pd.DataFrame(columns=[column])
        elif isinstance(df, (pd.DataFrame, pd.Series)):
            if hasattr(df, column):
                self.df = df
            else:
                raise AttributeError(f"Dataframe must have column {column}.")
        else:
            try:
                self.df = pd.DataFrame(data=df[column], columns=[column])
            except AttributeError as err:
                raise AttributeError(f"Dataframe must have column {column}.") from err

        if (
            not self.df.empty
            and not self.df[column].apply(lambda x: isinstance(x, Path)).all()
        ):
            self.df[column] = self.df[column].astype("string").map(Path)

        if exists:
            mask = ~self.df[column].apply(lambda x: x.exists())
            non_existing = self.df[column][mask]
            if any(mask):
                raise FileExistsError(
                    f"The following files do not exist: {non_existing}"
                )

    def __iter__(self) -> Iterable[tuple[Any, ...]]:
        return self.df.itertuples(name="Files")

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.df.iloc[index]
        else:
            return Files(directory=self.directory, df=self.df.iloc[index])

    @classmethod
    def concatenate(cls, files=None, directory=None, exists=True) -> Files:
        """
        Concatenate the file lists from multiple File instances
        and set the base directory without further action.

        Parameters
        ----------
        files : Iterable[Files] | None
            sequence with File instances
        directory: str | bytes | os.PathLike | None
            new base directory
        exists : bool
            raise `FileExistsError` if file in files does not exist

        Returns
        -------
        Files
        """
        if files is None:
            return cls()
        else:
            df = pd.concat([files_.df for files_ in files], join="inner")
            df = df.drop_duplicates().reset_index(drop=True)
            column = df.columns[0]
            return cls(df=df, directory=directory, exists=exists, column=column)

    @classmethod
    def from_path(
        cls,
        files=None,
        directory=None,
        column="file_path",
    ) -> Files:
        """
        Instantiate `Files` from a collection of file paths.

        Parameters
        ----------
        files : Iterable[str | bytes | os.PathLike] | str | bytes | os.PathLike | None
            sequence with File instances
        directory : str | bytes | os.PathLike
            new base directory
        column : str
            Name of column in `Files.df` carrying these files

        Returns
        -------
        Files
        """
        if isinstance(files, (str, bytes, os.PathLike)):
            df = pd.DataFrame(data=[files], columns=[column])
        elif isinstance(files, Iterable):
            df = pd.DataFrame(data=files, columns=[column])
        else:
            df = files
        return cls(directory=directory, df=df, column=column)

    @classmethod
    def from_glob(
        cls,
        directory=None,
        pattern="*.txt",
        regex=None,
        column="file_path",
    ) -> Files:
        """
        Instantiate `Files` from a search with glob and/or regex patterns.

        Parameters
        ----------
        pattern : str | None
            glob pattern passed to :func:`Path.glob`
        regex : str | None
            regex pattern passed to :func:`re.search` and applied in addition
            to glob pattern
        directory : str | bytes | os.PathLike
            new base directory in which to search
        column : str
            Name of column in `Files.df` carrying these files

        Returns
        -------
        Files
        """
        if directory is None:
            directory = Path().cwd()

        files = directory.glob(pattern)

        if regex is None:
            files = list(files)
        else:
            regex_ = re.compile(regex)
            files = [file_ for file_ in files if regex_.search(str(file_)) is not None]

        df = pd.DataFrame(data=files, columns=[column])
        return cls(directory=directory, df=df, column=column)

    def add_glob(
        self,
        pattern: str = "*.txt",
        regex=None,
        column: str = "other_file_path",
    ) -> Self:
        """
        Search for file paths using glob and/or regex pattern in base directory
        and provide files in new `column`.

        A logging.warning is given if the number of found files and those
        in `self.df` are different.

        Parameters
        ----------
        pattern : str | None
            glob pattern passed to :func:`Path.glob`
        regex : str | None
            regex pattern passed to :func:`re.search` and applied in addition
            to glob pattern
        column : str
            Name of column in `Files.df` carrying these files

        Returns
        -------
        Self
        """
        files = self.directory.glob(pattern)

        if regex is None:
            files = list(files)
        else:
            regex_ = re.compile(regex)
            files = [file_ for file_ in files if regex_.search(str(file_)) is not None]

        self.df[column] = pd.Series(files)
        if len(self.df) != self.df[column].count():
            logger.warning("Not all files are matched.")
        return self

    def exclude(
        self,
        stoplist=None,
        column="file_path",
        column_stoplist="file_path",
    ) -> Self:
        """
        Exclude files in `self.df.column` according to stoplist.

        Parameters
        ----------
        stoplist : Files | Iterable[bool | str | bytes | os.PathLike] | None
            Files to be excluded
        column : str
            key/column in `df` from which to exclude files
        column_stoplist : str
            key/column in `stoplist` from which to take files

        Returns
        -------
        Self
        """
        if stoplist is None or bool(stoplist) is False:
            pass
        elif not isinstance(stoplist, Files) and all(
            isinstance(item, bool) for item in stoplist
        ):
            selection = [not item_ for item_ in stoplist]
            self.df = self.df[selection]
        else:
            if isinstance(stoplist, Files):
                stoplist = stoplist.df[column_stoplist].astype("string")
            else:
                try:
                    stoplist = [str(item_[column_stoplist]) for item_ in stoplist]
                except TypeError:
                    stoplist = [str(item_) for item_ in stoplist]

            if len(stoplist) != 0:
                conditions = [
                    self.df[column].astype("string").str.contains(item_, regex=False)
                    for item_ in stoplist
                ]
                mask = functools.reduce(lambda x, y: x | y, conditions)
                self.df = self.df[~mask]
        return self

    def match_files(
        self,
        files,
        column="file_path",
        other_column="other_file_path",
    ) -> Self:
        """
        Add files in new column.

        A logging.warning is given if the number of files and those
        in `self.df` are different.

        Parameters
        ----------
        files : pd.Series
            New file list
        column : str
            Name of column in `Files.df` carrying files to match
        other_column : str
            Name of new column carrying files

        Returns
        -------
        Self
        """
        self.df[other_column] = files
        if self.df[column].count() != self.df[other_column].count():
            logger.warning("Not all files are matched.")
        return self

    def match_file_upstream(
        self,
        column: str = "file_path",
        pattern: str = "*.toml",
        regex=None,
        directory: str | bytes | os.PathLike = None,
        other_column="metadata",
    ) -> Self:
        """
        Find a matching file by applying :func:`locan.find_file_upstream` on
        each file in `self.df[column]`.

        Parameters
        ----------
        column : str
            Name of column in `Files.df` carrying files to match
        pattern : str | None
            glob pattern passed to :func:`Path.glob`
        regex : str | None
            regex pattern passed to :func:`re.search` and applied in addition
            to glob pattern
        directory :
            top directory in which to search
        other_column : str
            Name of new column carrying files

        Returns
        -------
        Self
        """
        matched_file = [
            find_file_upstream(
                sub_directory=file_,
                pattern=pattern,
                top_directory=directory,
                regex=regex,
            )
            for file_ in self.df[column]
        ]
        self.df[other_column] = matched_file
        return self

    def print_summary(self) -> None:
        """
        Print summary of Files.

        Returns
        -------
        None
        """
        print(f"Number of files: {len(self.df)}")
        print(f"Base directory: {self.directory}")
        print(f"Columns: {self.df.columns}")
        print(self.df.describe().loc[["count", "unique"]])

    def group_identifiers(self):
        """
        Get categories defined in self.df.group.

        Returns
        -------
        pd.
        """
        return self.df.group.cat.categories

    def grouped(self) -> pd.core.groupby.DataFrameGroupBy:
        """
        Get groupby instance based on group_identifiers.

        Returns
        -------
        pandas.core.groupby.DataFrameGroupBy
        """
        return self.df.groupby(by="group")

    def set_group_identifier(
        self, name=None, pattern=None, glob=None, regex=None, column="file_path"
    ) -> Self:
        """
        Set group_identifier `name` for files in `column` as identified by
        string pattern and/or glob pattern and/or regex
        and keep them in column "group".

        Parameters
        ----------
        name : str
            new group_identifier
        pattern : str | None
            string pattern
        glob : str | None
            glob pattern passed to :func:`Path.match`
        regex : str | None
            regex pattern
        column : str
            Name of column in `Files.df` carrying files to match

        Returns
        -------
        Self
        """
        if all(key_ is None for key_ in [pattern, glob, regex]):
            return self

        if pattern is not None:
            self.df["_mask_pattern"] = (
                self.df[column].astype("string").str.contains(pattern, regex=False)
            )

        if glob is not None:
            self.df["_mask_glob"] = self.df[column].apply(lambda x: x.match(glob))

        if regex is not None:
            self.df["_mask_regex"] = (
                self.df[column].astype("string").str.contains(regex, regex=True)
            )

        mask_columns = [
            key_
            for key_ in ["_mask_pattern", "_mask_glob", "_mask_regex"]
            if key_ in self.df.columns
        ]

        if len(mask_columns) == 1:
            self.df["mask"] = self.df[mask_columns]
        else:
            self.df["mask"] = self.df[mask_columns].all(axis=1)

        if "group" not in self.df.columns:
            self.df["group"] = pd.NA
            self.df["group"] = self.df["group"].astype("category")
        else:
            if self.df.loc[self.df["mask"], "group"].notna().any():
                logger.warning(f"Previously defined groups are overwritten with {name}")

        if name not in self.df["group"].cat.categories:
            self.df["group"] = self.df["group"].cat.add_categories([name])
        self.df.loc[self.df["mask"], "group"] = name
        self.df = self.df.drop(columns=mask_columns + ["mask"])

        if self.df.group.isna().any():
            logger.info("Some group identifiers are still NAN.")
        return self