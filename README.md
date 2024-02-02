# stare_pet

This python-based implementation of Source-to-Target Automated Rotating
Estimation (STARE) is based on
[Bartlett, et al. 2022](https://doi.org/10.1016/j.neuroimage.2022.118901)
and the original
[matlab-based implementation](https://github.com/elizabeth-bartlett/STARE)
.

Briefly, STARE allows for estimating an input function rather than collecting
it from blood when modeling positron emission tomography (PET) kinetics.
Details are in [the paper](https://doi.org/10.1016/j.neuroimage.2022.118901)
and [preprint](https://doi.org/10.1101/2021.09.15.460504).

Documentation is being written, as is the code. Current incomplete docs
are available via [github pages](https://mfschmidt.github.io/stare_pet/).

## Installation

### Straight Docker

The fastest way to just run __stare__ on your data is to 
use the fully prepared docker image. Assuming docker is installed on your
machine, you can get it from docker hub and run it like so.

    docker pull mfschmidt/stare:latest

    docker run -it --rm \
    -v /home/me:/home/me:rw \
    mfschmidt/stare:latest sub-01 \
    --input-path /home/me/my_pet_data \
    --output-path /home/me/my_stare_output

There are several reasons this may not be the best solution for you, so
more complex solutions are available to resolve some common issues.

### Docker as a normal user

Straight docker will run as root, and you may not have the ability to change
ownership of all the root-owned files after __stare__ has completed. A script is
supplied to handle this user mapping, which would require downloading the
github repository and running like so. Obviously, change the paths to match
your own situation.

    docker pull mfschmidt/stare:latest
    cd /home/me
    git clone https://github.com/mfschmidt/stare_pet
    cd /home/me/stare_pet

    ./stare-docker sub-01 \
    --input-path /home/me/my_pet_data \
    --output-path /home/me/my_stare_output

### Running locally

The biggest impediment to running locally, as opposed to inside the docker
container, is that __stare__ is dependent on external software,
[pet_pvc](https://github.com/UCL/PETPVC) for partial volume correction.
That software is pre-installed into the docker container, 
but you'll need to install it on any machine you'd like to use to
run __stare__ locally. See their instructions, or copy commands from inside 
__stare_pet__'s Dockerfile to do so.

After __pet_pvc__ is installed, you can set up a python virtual environment,
install stare's required dependencies, then install __stare_pet__ itself, and
you can then run locally. As before, change paths to suit your preferences
and situation.

    # Create, activate, and set up a python virtual environment.
    python3 -m venv /home/me/.virtualenvs/stare
    source /home/me/.virtualenvs/stare/bin/activate
    pip install git+https://github.com/mfschmidt/stare_pet.git#egg=stare_pet

    stare sub-01 \
    --input-path /home/me/my_pet_data \
    --output-path /home/me/my_stare_output

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

    stare --options-file my_stare_settings.json

You can get more options and documentation with help.

    stare --help

### Jupyter Notebook

From within your python virtual environment, with __stare_pet__ already
installed, you may additionally install packages to host a jupyter notebook
and use __stare_pet__ code from there.

    pip install notebook jupyter jupyterlab
    cd /home/me/stare_pet
    jupyter lab

See this repository's 'examples/dev_01_description_of_stare.ipynb'.

### Function calls

See how the jupyter notebook runs each step or explore the `examples/` code and
documentation.
