# stare_pet

This python-based implementation of Source-to-Target Automated Rotating
Estimation (STARE) is based on
[Bartlett, et al. 2022](https://doi.org/10.1016/j.neuroimage.2022.118901)
and the original
[matlab-based implementation](https://github.com/elizabeth-bartlett/STARE)
.

Briefly, STARE allows for estimating input data rather than collecting
it from blood when modeling positron emission tomography (PET) kinetics.
Details are in [the paper](https://doi.org/10.1016/j.neuroimage.2022.118901)
and [preprint](https://doi.org/10.1101/2021.09.15.460504).

Documentation is being written, as is the code. Current incomplete docs
are available via [github pages](https://mfschmidt.github.io/stare_pet/).

## Installation

stare_pet is made available via pypi or this github repository.

    pip install stare_pet

or

    pip install git+https://github.com/mfschmidt/stare_pet.git#egg=stare_pet

## Usage

There are three ways to use the *stare_pet* library. 1. Command-line usage
is nice to simply execute the pipeline on properly formatted data. 2.
Jupyter notebook usage is nice to execute a step at a time and understand
what's happening at each step. 3. Calling library functions directly (as
demonstrated in 1 and 2) allows you to take what you need for your own
analyses.

### Command line

To execute the stare pipeline with all options specified on the command-line:

    stare --input-path /mnt/data/stuff --output-path . --verbose

Or, you may want to save your settings in a file and have stare get options
from there.

    stare --file my_stare_settings.json

You can get more options and documentation with help.

    stare --help

### Jupyter Notebook

See this repository's 'examples/dev_01_description_of_stare.ipynb'.

### Function calls

See how the jupyter notebook runs each step or explore the code and
documentation.