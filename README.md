
![misfits's interface](https://github.com/peppedilillo/misfits/blob/main/assets/misfits.gif?raw=true)

# misfits

Misfits is a FITs table viewer for the terminal, written in python.
I want it to be snappy as hell and fully usable without touching the mouse.
It currently has some limitations (e.g. won't display VLA columns), but will work on them eventually.
It leverages astropy and pandas, and is built using [textual](https://www.textualize.io/).
Works on Linux, macOS and Windows. Performances on Windows are worse.
Renders best on modern terminals.

### Installation

#### Installing with `pip`

`pip install misfits`

Make sure to be installing into a fresh python>=3.11 environment!

#### Installing with `uv`

`uv tool install misfits`

With the other methods, you are supposed to activate the misfits environment first to use it.
Installing with uv you won't need that, and you will be able to call misfits from terminal with one line: `misfits`.
If you are unsure about uv: don't, give it a [try](https://docs.astral.sh/uv/getting-started/installation/)!
It is a great package manager from the people behind ruff and other python tools.

#### Installing with anaconda

`conda env create -f conda-env.yml`

Will create a new environment and install `misfits` in it.

### Usage

From the terminal, run `misfits path_to_file.fits`, `misfits .`, or simply `misfits`. 

### Contributing

Found a bug? Want a feature? Open an issue, a PR, or post in the discussion section of this repo.