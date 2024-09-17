
![misfits's interface](https://github.com/peppedilillo/misfits/blob/main/assets/misfits.gif?raw=true)

# misfits

Misfits is a FITs table viewer for the terminal, written in python.
I want it to be snappy as hell and fully usable without touching the mouse.
It currently has some limitations (e.g. won't display array or VLA columns), but will work on them eventually.
It leverages astropy and pandas, and is built using [textual](https://www.textualize.io/).
Works on Linux, macOS and Windows. Performances on Windows are worse.
Renders best on modern terminals: windows new terminal, macOS iTerm2, your typical linux distro is likely fine as it is.

### Installation

`pip install misfits`

Make sure to be installing into a fresh python 3.11 environment!

#### Installing with uv

`uv tool install misfits`

If you like misfits, this is probably the best method. But you should install uv first, see the [docs](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11 is also required.
With the other methods, you are supposed to activate the misfits environment to use it.
This won't, and you will be able to call misfits from terminal with one line: `misfits .`.

If you are unsure about `uv`: don't, give it a try! 
It is a great package manager from the [people](https://astral.sh/) behind ruff and other python tools.

#### Installing with anaconda

`conda env create -f conda-env.yml`

Will create a new environment and install `misfits` in it.

### Usage

From the terminal, type `misfits path_to_file.fits` or `misfits .`. 
The latter will open a prompt to choose a fits file from your current directory.
