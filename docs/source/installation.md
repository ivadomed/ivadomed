# Installation

`ivadomed` requires Python >= 3.6 and PyTorch >= 1.5.0. We recommend
working under a virtual environment, which could be set as follows:

```bash
virtualenv venv-ivadomed --python=python3.6
source venv-ivadomed/bin/activate
```

## Install from release (recommended)

Install `ivadomed` and its requirements from [Pypi](https://pypi.org/project/ivadomed/):

```bash
pip install --upgrade pip
pip install ivadomed
```

## Install from source

Bleeding-edge developments are available on the project's master branch on Github.
Installation procedure is the following:

```bash
git clone https://github.com/neuropoly/ivado-medical-imaging.git
cd ivado-medical-imaging
pip install -e .
```
