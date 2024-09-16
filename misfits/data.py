from enum import Enum
import re

from pathlib import Path

from astropy.io import fits
from astropy.io.fits import BinTableHDU
from astropy.io.fits import TableHDU


def is_table(hdu: fits.FitsHDU):
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
    """Return relevant informations on column type"""
    count, type_char, matchA = parse_format(tform)
    if type_char[0] in "PQ":
        return ColumnType.VARLEN
    elif count == 1:
        return ColumnType.SCALAR
    else:
        return ColumnType.VECTOR


async def get_fits_content(fits_path: str | Path) -> tuple[dict]:
    """Retrieves content from a FITS file and stores it in a tuple dict.
    Each tuple's records referes to one FITS HDU. CPU-heavy."""
    content = []
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            ist = is_table(hdu)
            if ist:
                data = hdu.data
                cscalar, cvector, cvarlen = [], [], []
                for col in data.columns:
                    match get_column_type(col.format):
                        case ColumnType.SCALAR:
                            cscalar.append(col.name)
                        case ColumnType.VECTOR:
                            cvector.append(col.name)
                        case ColumnType.VARLEN:
                            cvarlen.append(col.name)

            content.append(
                {
                    "name": hdu.name,
                    "type": hdu.__class__.__name__,
                    "header": dict(hdu.header) if hdu.header else None,
                    "is_table": ist,
                    "data": hdu.data if ist else None,
                    "columns_scalar": cscalar if ist else None,
                    "columns_varlen": cvarlen if ist else None,
                    "columns_vector": cvector if ist else None,
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
