from enum import Enum


class ColumnType(Enum):
    SCALAR = 0
    VECTOR = 1
    VARLEN = 2


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
