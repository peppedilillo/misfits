from collections import OrderedDict
from enum import Enum
import re

from pathlib import Path

from astropy.io import fits
from astropy.io.fits import BinTableHDU
from astropy.io.fits import TableHDU
from astropy.io.fits import FITS_rec
import pandas as pd
import numpy as np


def is_table(hdu: fits.FitsHDU):
    """Check whether and HDU contains table data."""
    return type(hdu) in [TableHDU, BinTableHDU]


class ColumnType(Enum):
    SCALAR = 0
    VECTOR = 1
    VARLEN = 2


def parse_format(tform: str) -> tuple[int, str, str]:
    """
    Based on  Starlink (STIL) Java implementation, see
    `https://github.com/Starlink/starjava/blob/master/fits/src/main/uk/ac/starlink/fits/ColumnReader.java`

    :param tform: a FITS TFORM string
    :return: an integer representing the column length (1 for scalar columns, 2+ for vector columns),
    a string, representing the type of the column values.
    a string, representing additional information used for interpreting vector and
    variable length columns.
    """
    pattern = r'([0-9]*)([LXBIJKAEDCMPQ])(.*)'
    match = re.match(pattern, tform)
    if not match:
        # TODO: Handle this error
        raise ValueError(f"Error parsing TFORM value {tform}")
    scount = match.group(1)
    type_char = match.group(2)
    matchA = match.group(3).strip()
    count = 1 if scount == '' else int(scount)
    return count, type_char, matchA


def get_column_type(tform: str) -> ColumnType:
    """Classifies a column based on its content."""
    count, type_char, matchA = parse_format(tform)
    # "PQ" type char is used for identifying variable len columns.
    if type_char[0] in "PQ":
        return ColumnType.VARLEN
    elif count == 1:
        return ColumnType.SCALAR
    else:
        return ColumnType.VECTOR


async def get_fits_content(fits_path: str | Path) -> tuple[dict]:
    """Retrieves content from a FITS file and stores it in a tuple dict.
    Each tuple's records referes to one FITS HDU.
    Can take some time for large tables"""
    content = []
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            content.append(
                {
                    "name": hdu.name,
                    "type": hdu.__class__.__name__,
                    "header": dict(hdu.header) if hdu.header else None,
                    "is_table": (ist := is_table(hdu)),
                    "data": hdu.data if ist else None,
                }
            )
    return tuple(content)


FITS_SIGNATURE = b"SIMPLE  =                    T"


def _validate_fits(filepath: Path) -> bool:
    """Checks if a file is a FITS."""
    # Following the same approach of astropy.
    try:
        with open(filepath, "rb") as file:
            # FITS signature is supposed to be in the first 30 bytes, but to
            # allow reading various invalid files we will check in the first
            # card (80 bytes).
            simple = file.read(80)
    except OSError:
        return False
    match_sig = simple[:29] == FITS_SIGNATURE[:-1] and simple[29:30] in (b"T", b"F")
    return match_sig


class DataContainer:
    """This class manages FITS table data.
    It makes data representable on the user machine, determines which columns to show,
    dispatch table entries and deals with dataset queries, when queries are possible."""

    def __init__(self, records: FITS_rec):
        self.records: fits.FITS_rec | pd.DataFrame = records
        self._len = len(records)
        self.columns = {col.name: get_column_type(col.format) for col in records.columns}
        self.displayable_columns = [colname for colname, coltype in self.columns.items() if coltype is not ColumnType.VARLEN]
        self.can_promote = all([coltype != ColumnType.VARLEN for coltype in self.columns.values()])
        self.promoted = False
        self.mask: None | pd.Index = None

    def __len__(self):
        """Returns length of possibly filtered dataset."""
        return self._len

    @staticmethod
    def maybe_correct_endianess(records: FITS_rec):
        """Convert FITS records to user machine endiannes if they differ."""
        if not records.dtype.isnative:
            records = records.byteswap().view(records.dtype.newbyteorder("="))
        return records

    def promote(self):
        """Promote a fits records table to a proper dataframe."""
        assert self.can_promote
        self.promoted = True
        self.records = self._to_pandas(self.records)  # Table(self.records).to_pandas()
        self.mask = self.records.index

    # table gets promoted when first converted to dataframe.
    # this enables the usage of pandas queries. promotion happens at first filter call.
    # a promoted table cannot be demoted.
    def query(self, query: str):
        """Leverage pandas queries to filter dataset"""""
        if not self.can_promote:
            raise ValueError("Trying to filter an unpromotable table")
        if not self.promoted:
            self.promote()
        filtered_df = self.records.query(query) if query else self.records
        self.mask = filtered_df.index
        self._len = len(filtered_df)

    def get_rows(self, slice):
        """Dispatch rows based on slice."""
        if self.promoted:
            table_slice = self.records.iloc[self.mask[slice]]
        else:
            # this looks eccentric but is faster than list comprehension (WHY?) and
            # has the benefit of having homogenous formatting with promoted table
            table_slice = self._to_pandas(self.records[slice])
        return table_slice.itertuples(index=False)

    def get_columns(self):
        """Dispatch table's displayable columns"""
        return self.displayable_columns

    def _to_pandas(self, table):
        """Transforms a fits records table into a dataframe"""
        out = OrderedDict()
        for colname in self.displayable_columns:
            column = self.maybe_correct_endianess(table[colname])
            if self.columns[colname] is ColumnType.VECTOR:
                if column.dtype.kind == "f":
                    # this is a workaround to Textual not applying cell formatting
                    # recursively. TODO: improve this logic?
                    column = np.round(column, 2)
                column = column.tolist()
            out[colname] = column

        df = pd.DataFrame(out, index=None)
        return df
