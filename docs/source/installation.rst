Installation
============

``ivadomed`` requires Python >= 3.6 and <3.9  as well as PyTorch == 1.5.0. We recommend
working under a virtual environment, which could be set as follows:

::

    virtualenv venv-ivadomed
    source venv-ivadomed/bin/activate

Install from release (recommended)
----------------------------------

Install ``ivadomed`` and its requirements from
`Pypi <https://pypi.org/project/ivadomed/>`__:

::

    pip install --upgrade pip
    pip install ivadomed

Install from source
-------------------

Bleeding-edge developments are available on the project's master branch
on Github. Installation procedure is the following:

::

    git clone https://github.com/neuropoly/ivadomed.git
    cd ivadomed
    pip install -e .

