Installation
============

Supported OS
------------

Currently, we only support ``MacOS`` and ``Linux`` operating systems. ``Windows``
users have the possibility to install and use ``ivadomed`` via
`Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/>`_.For MacOs users, we strongly recommend to follow the bellow steps before the installation.

Open your bash/zsh script file with editor on your computer.

    If you are using bash shell
    ::

        vim ~/.bashrc

    If you are using zsh shell
    ::
        vim ~/.zshrc

Write in your .bashrc/.zshrc file with following line.

::

    export HDF5_USE_FILE_LOCKING='FALSE'

Save this change and restart your terminal to apply the change.

There are fundamentally three different approaches to install IvadoMed:

1) Install via Conda
    This is the easiest way for personal computers.
2) Install via Venv/VirtualEnv
    This is compatible with ComputeCanada cluster environment.
3) Install via Docker
    This is when you already have Docker ready and just want to run simple non GPU accelerated IvadoMed Commands.

Approach 0: Conda
===================
Step 1.0: Create new Conda Env called IvadoMedEnv (may taken ~10 minutes)
---------------------------------------------------------------------------
::
    conda env create --file environment.yml

Step 2.0 : Activate environment and use
-------------------------------------------
::
    conda activate IvadoMedEnv

Note that this is NOT compatible with ComputeCanada because of their no anaconda policy: https://docs.computecanada.ca/wiki/Anaconda/en


Approach 1: Venv
===================

Step 1.0: Setup Python Virtual Environment.
---------------------------------------------------

``ivadomed`` requires Python >= 3.6 and <3.9. We recommend
working under a virtual environment, which could be set as follows:

::

    virtualenv venv-ivadomed
    source venv-ivadomed/bin/activate

.. warning::
   If the default Python version installed in your system does not fit the version requirements, you might need to specify a version of Python associated with your virtual environment:

   ::

     virtualenv venv-ivadomed --python=python3.6


Step 1.1: Install PyTorch 1.5 and TorchVision (CPU)
---------------------------------------------------
::
    pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.5.0+cpu torchvision==0.6.0+cpu

Optional Alternative Step 1.1: Install PyTorch 1.5 and TorchVision (GPU)
---------------------------------------------------
If you have a GPU and it has CUDA already setup etc, do the follow or use whatever CUDA version you have.
::
    pip install torch torchvision


Step 2: Install from release (recommended)
----------------------------------

Install ``ivadomed`` and its requirements from
`Pypi <https://pypi.org/project/ivadomed/>`__:

::

    pip install --upgrade pip
    pip install ivadomed

Alternative Step 2: Install from source
-------------------

Bleeding-edge developments are available on the project's master branch
on Github. Installation procedure is the following:

::

    git clone https://github.com/neuropoly/ivadomed.git
    cd ivadomed
    pip install -e .


Install pre-commit hooks for development
----------------------------------------

We use ``pre-commit`` to enforce a limit on file size.
After you've installed ``ivadomed``, install the hooks:

::

    pre-commit install

