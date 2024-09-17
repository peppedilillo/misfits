from datetime import datetime
from enum import Enum

from misfits.data import ColumnType


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2


class Logger:
    def __init__(self):
        self.logstack = []

    def push(self, message: str, level: LogLevel | None = LogLevel.INFO):
        now_str = "[dim cyan]" + datetime.now().strftime("(%H:%M:%S)") + "[/]"
        match level:
            case LogLevel.INFO:
                prefix = f"{now_str} [dim green][INFO][/]: "
            case LogLevel.WARNING:
                prefix = f"{now_str} [dim yellow][WARNING][/]: "
            case LogLevel.ERROR:
                prefix = f"{now_str} [bold red][ERROR][/]: "
            case _:
                prefix = ""
        self.logstack.append(prefix + message)

    def pop(self) -> str | None:
        return self.logstack.pop(0) if self.logstack else None

    def push_hducontent(self, content):
        # fmt: off
        self.push(f"Found HDU {repr(content['name'])} of type {repr(content['type'])}")
        if content["is_table"]:
            ncols = len(content["data"].columns)
            self.push(f"HDU contains a table with {len(content['data'])} rows and {ncols} columns")
        # fmt: on

    def push_datacontent(self, data):
        # fmt: off
        columns = {coltype: [] for coltype in ColumnType}
        for colname, coltype in data.columns.items():
            columns[coltype].append(colname)

        if scalar_columns := columns[ColumnType.SCALAR]:
            self.push(f"Data table contains {len(scalar_columns)} scalar columns: {scalar_columns}")
        if vector_columns := columns[ColumnType.VECTOR]:
            self.push(f"Data table contains {len(vector_columns)} vector columns: {vector_columns}")
        if varlen_columns := columns[ColumnType.VARLEN]:
            self.push(f"Data table contains {len(varlen_columns)} variable len columns: {varlen_columns}", level = LogLevel.WARNING)
        # fmt: on


log = Logger()
