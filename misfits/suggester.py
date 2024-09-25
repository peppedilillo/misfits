from pathlib import Path
import itertools

from textual.suggester import Suggester


def is_directory_or_fitsfile(path: Path):
    return path.is_dir() or path.suffix == ".fits"


class PathSuggester(Suggester):
    """Suggest either directories of file with `fits` extension.
    This class is based on an original implementation by Will McGugan, https://github.com/willmcgugan.
    The code was not released as part of a package but provided via private communication.
    It comes with no license.
    Thank you very much for letting me use this, Will!
    TODO: remove once the suggester will be released with textual.
    """

    def __init__(self):
        super().__init__(case_sensitive=True)

    async def get_suggestion(self, value: str) -> str | None:
        """Suggest the first matching directory"""
        try:
            path = Path(value)
            if is_directory_or_fitsfile(path):
                return None
            name = path.name
            possible_paths = [
                str(sibling_path)
                for sibling_path in itertools.islice(
                    path.parent.expanduser().iterdir(), 100
                )
                if sibling_path.name.lower().startswith(name.lower()) and is_directory_or_fitsfile(sibling_path)
            ]
            if possible_paths:
                possible_paths.sort(key=str.__len__)
                suggestion = possible_paths[0]
                if "~" in value:
                    home = str(Path("~").expanduser())
                    suggestion = suggestion.replace(home, "~")
                return suggestion

        except FileNotFoundError:
            pass
        return None
