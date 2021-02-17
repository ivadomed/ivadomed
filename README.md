  
![ivadomed Overview](https://raw.githubusercontent.com/ivadomed/doc-figures/main/index/overview_title.png)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02868/status.svg)](https://doi.org/10.21105/joss.02868)
[![Coverage Status](https://coveralls.io/repos/github/ivadomed/ivadomed/badge.svg?branch=master)](https://coveralls.io/github/ivadomed/ivadomed?branch=master)
[![test status](https://github.com/ivadomed/ivadomed/workflows/Run%20tests/badge.svg)](https://github.com/ivadomed/ivadomed/actions?query=workflow%3A%22Run+tests%22)
[![publish package](https://github.com/ivadomed/ivadomed/workflows/Publish%20Package/badge.svg)](https://github.com/ivadomed/ivadomed/actions?query=workflow%3A%22Publish+Package%22)
[![Documentation Status](https://readthedocs.org/projects/ivado-medical-imaging/badge/?version=stable)](https://ivadomed.org/en/stable/?badge=stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Twitter Follow](https://img.shields.io/twitter/follow/ivadomed.svg?style=social&label=Follow)](https://twitter.com/ivadomed)

`ivadomed` is an integrated framework for medical image analysis with deep learning.

The technical documentation is available [here](https://ivadomed.org).

## Installation

``ivadomed`` requires Python >= 3.6 and < 3.9 as well as PyTorch == 1.5.0. We recommend working under a virtual environment, which could be set as follows:

```bash
virtualenv venv-ivadomed
source venv-ivadomed/bin/activate
```

### Install from release (recommended)

Install ``ivadomed`` and its requirements from `Pypi <https://pypi.org/project/ivadomed/>`__:

```bash
pip install --upgrade pip
pip install ivadomed
```

### Install from source

Bleeding-edge developments are available on the project's master branch
on Github. Installation procedure is the following:

```bash
git clone https://github.com/neuropoly/ivadomed.git
cd ivadomed
pip install -e .
```

## Contributors
<p float="left">
  <img src="https://raw.githubusercontent.com/ivadomed/doc-figures/main/contributors/neuropoly_logo.png" height="80" />
  <img src="https://raw.githubusercontent.com/ivadomed/doc-figures/main/contributors/mila_logo.png" height="80" />
  <img src="https://raw.githubusercontent.com/ivadomed/doc-figures/main/contributors/ivado_logo.png" height="80" />
</p>

This project results from a collaboration between the [NeuroPoly Lab](https://www.neuro.polymtl.ca/)
and the [Mila](https://mila.quebec/en/). The main funding source is [IVADO](https://ivado.ca/en/).

[List of contributors](https://github.com/neuropoly/ivadomed/graphs/contributors)
