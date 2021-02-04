Installation
============

Supported OS
------------

Currently, we only support ``MacOS`` and ``Linux`` operating systems. ``Windows``
users have the possibility to install and use ``ivadomed`` via
`Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/>`_.

Python
------

``ivadomed`` requires Python >= 3.6 and <3.9. We recommend
working under a virtual environment, which could be set as follows:

::

    virtualenv venv-ivadomed
    source venv-ivadomed/bin/activate

.. warning::
   If the default Python version installed in your system does not fit the version requirements, you might need to specify a version of Python associated with your virtual environment:

   ::

     virtualenv venv-ivadomed --python=python3.6



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
