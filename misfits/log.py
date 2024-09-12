from datetime import datetime
from enum import Enum


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

    def push_fitcontents(self, content):
        # fmt: off
        self.push(f"Found HDU {repr(content['name'])} of type {repr(content['type'])}")
        if content["is_table"]:
            ncols = len(content["data"].columns)
            self.push(f"HDU contains a table with {len(content['data'])} rows and {ncols} columns")
            if content["columns_scalar"]:
                self.push(f"Columns {content['columns_scalar']} are scalar.")
            if content["columns_vector"]:
                self.push(f"Columns {content['columns_vector']} are vectorial.")
            if content["columns_varlen"]:
                self.push(f"Columns {content['columns_varlen']} are variable length arrays.")
        # fmt: on


log = Logger()
