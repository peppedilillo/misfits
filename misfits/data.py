from pathlib import Path
from typing import Hashable

from astropy.io import fits
from astropy.io.fits import BinTableHDU
from astropy.io.fits import TableHDU
import numpy as np
from pandas import Series
from pandas.core.computation.eval import eval as _eval


def get_cleaned_column_resolvers(arr) -> dict[Hashable, Series]:
    """
    Return the special character free column resolvers of a dataframe.

    Column names with special characters are 'cleaned up' so that they can
    be referred to by backtick quoting.
    Used in :meth:`DataFrame.eval`.
    """
    from pandas.core.computation.parsing import clean_column_name

    return {
        clean_column_name(k.name): arr[k.name]
        for k in arr.columns
        if not isinstance(k, int)
    }


def filter_array(query: str, arr: fits.FITS_rec) -> fits.FITS_rec:
    if query == "":
        return arr
    mask = _eval(query, resolvers=(get_cleaned_column_resolvers(arr),))
    if mask.dtype != np.dtype("bool") or mask.shape != (len(arr),):
        raise ValueError()
    return arr[mask]


async def get_fits_content(fits_path: str | Path) -> tuple[dict]:
    """Retrieves content from a FITS file and stores it in a tuple dict.
    Each tuple's records referes to one FITS HDU. CPU-heavy."""

    def is_table(hdu: fits.FitsHDU):
        return type(hdu) in [TableHDU, BinTableHDU]

    def single_columns(hdu: fits.FitsHDU):
        return [c.name for c in hdu.data.columns if len(c.array.shape) <= 1]

    def array_columns(hdu: fits.FitsHDU):
        return [c.name for c in hdu.data.columns if len(c.array.shape) > 1]

    def variable_len_columns(hdu: fits.FitsHDU):
        return [c.name for c in hdu.data.columns if not ("Q" in c.format or "P" in c.format)]

    content = []
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            content.append({
                "name": hdu.name,
                "type": hdu.__class__.__name__,
                "header": dict(hdu.header) if hdu else None,
                "is_table": (ist := is_table(hdu)),
                "data": hdu.data if ist else None,
                "columns": single_columns(hdu) if ist else None,
                "columns_varlen": variable_len_columns(hdu) if ist else None,
                "columns_arrays": array_columns(hdu) if ist else None,
            })
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
