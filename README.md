# OCT
`OCT_lib.py`

A library to read and manipulate OCT images taken during suction-experiments. Initially developed for an Axsun OEM OCT.
It defines a `Bscan` Class with which the most common OCT operation can be performed. Notebooks in folder `01-ExampleNotebooks` show how to read-in the raw data from the OCT, do basic postprocessing, find the surface and its height, and compare unloaded and loaded conditions. 

- [OCT](#oct)
- [Installation](#installation)
  - [1. `pip` package installation](#1-pip-package-installation)
  - [2. Installation of dependencies:](#2-installation-of-dependencies)
- [Example Scripts](#example-scripts)
- [Example Datasets](#example-datasets)
- [License](#license)
# Installation

## 1. `pip` package installation
To install with `pip`, simply copy-paste the following command in the terminal (after having activated any appropriate virtual environment):\
`pip install git+https://github.com/LucaBartolini/OCT`
## 2. Installation of dependencies:
If you want to install all required libraries, simply run: \
`pip install -r requirements.txt` \
[As general best-practice, it is recommended to install all the dependencies in a virtual environment.]

# Example Scripts
In folder `01-ExampleScripts`, there are three Jupyter Notebooks:
- `01-RAWdata_to_Results.ipynb`: to read raw data from the OCT, apply post-processing, find surface, and store processed data
- `02-Relaxed-VS-Suction_all_info_PNG.ipynb`: to manipulate and plot the processed data
- `zz-Pressure_Setpoints_generator.ipynb`: to create an arbitrary pressure profile in function of time $P(t)$, that will be imposed during a suction experiment.

# Example Datasets
- `PDMS_phantom`: measurements in 3 regions (`boundary`, `thick`, `thin`), under two loading conditions (`rel` unloaded, and `suc` under 500mbar suction). Each repeated 5 times
- `Thumb` : the author's thumb, in two conditions
`-rel` (unloaded) and `-suc` (under suction), each with two images. From dataset `20190708_CScanThumb`


# License
`OCT_lib` follows the MIT License; anyone can freely use.