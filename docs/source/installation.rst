Installation
============

Supported OS
------------

Currently, we only support ``MacOS`` and ``Linux`` operating systems. ``Windows``
users have the possibility to install and use ``ivadomed`` via
`Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/>`_. The steps below (about updating bashrc) are strongly recommended for MacOS users in the installation process but are optional for Linux users.

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

There are fundamentally two different approaches to install ``IvadoMed``:

1) Install via Conda
    This is the easiest way for personal computers.

2) Install via Venv/VirtualEnv
    This is compatible with ComputeCanada cluster environment.

Approach 1: Conda
------------------

Step 1: Clone the `ivadomed <https://github.com/ivadomed/ivadomed>`_ repository.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    git clone https://github.com/neuropoly/ivadomed.git
    cd ivadomed

Step 2: Create new Conda Env called IvadoMedEnv (may take 5 to 15 minutes)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    conda env create --file environment.yml

Step 3 : Activate environment and use
++++++++++++++++++++++++++++++++++++++

::

    conda activate IvadoMedEnv

Step 4 : Install from source
++++++++++++++++++++++++++++++++++++++

::

    pip install -e .

Approach 2: Venv
-----------------

Step 1: Setup Python Virtual Environment.
+++++++++++++++++++++++++++++++++++++++++

``ivadomed`` requires Python >= 3.6 and <3.9 (If you are using `Compute Canada <https://www.computecanada.ca/>`_, you can load modules (e.g. python 3.9) as mentioned  `here <https://intranet.neuro.polymtl.ca/computing-resources/compute-canada#modules>`_ and `also here <https://docs.computecanada.ca/wiki/Utiliser_des_modules/en#Loading_modules_automatically>`_ ). We recommend
working under a virtual environment, which could be set as follows:

::

    virtualenv venv-ivadomed
    source venv-ivadomed/bin/activate


.. warning::
   If the default Python version installed in your system does not fit the version requirements, you might need to specify a version of Python associated with your virtual environment:

   ::

     virtualenv venv-ivadomed --python=python3.6

Step 2: Clone the `ivadomed <https://github.com/ivadomed/ivadomed>`_ repository.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


::

    git clone https://github.com/ivadomed/ivadomed.git
    cd ivadomed
 

Step 3: Install PyTorch 1.5 and TorchVision
+++++++++++++++++++++++++++++++++++++++++++

If you have a compatible NVIDIA GPU that supports CUDA, run the following command:

::
   
   pip install -r requirements_gpu.txt


If you do not have a compatible GPU, run the following installer to use ``ivadomed`` with CPU.


::

    pip install -r requirements.txt


Step 4: Install from release (recommended)
++++++++++++++++++++++++++++++++++++++++++

Install ``ivadomed`` and its requirements from
`Pypi <https://pypi.org/project/ivadomed/>`__:

::

    pip install --upgrade pip
    pip install ivadomed

(Optional) Alternative Step 4 for Developers: Install from source
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Bleeding-edge developments are available on the project's master branch
on Github. Installation procedure is the following at repository root:

::

    cd ivadomed
    pip install -e .


(Optional) Step 5 for Developers Install pre-commit hooks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We use ``pre-commit`` to enforce a limit on file size.
After you've installed ``ivadomed``, install the hooks:

::

    pip install -r requirements_dev.txt
    pre-commit install
