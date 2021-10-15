Installation
============

Supported OS
++++++++++++

Currently, we only support ``MacOS``, ``Linux`` and ``Windows`` (`Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/>`_ is not suppoorted.)

Step 0: MANDATORY Preparatory Step for Mac
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The steps below (about updating bashrc) are strongly recommended for MacOS users in the installation process but are optional for Linux/Windows  users.

Open your bash/zsh script file with editor on your Mac.

.. tabs::

    .. tab:: zsh (most recent macs)

        ::

            vim ~/.zshrc

    .. tab:: bash (older macs or if manually changed)

        ::

            vim ~/.bashrc


Write in your .bashrc/.zshrc file with following line. This is needed as OS will lock HDF5 generated during training and produce a rather obscure error.

::

    export HDF5_USE_FILE_LOCKING='FALSE'

Save this change and restart your terminal to apply the change.

Step 1: Setup dedicated python environment
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can setup ``IvadoMed`` using either Conda or Venv:

.. tabs::

    .. tab:: Install via Conda

        This is the easiest way for personal computers.

        1. Create new conda environment file

        ::

            conda env create --file environment.yml

        2. Activate the new conda environment (default named IvadoMedEnv)

        ::

            conda activate IvadoMedEnv

    .. tab:: Install via Venv/VirtualEnv

        This is compatible with ComputeCanada cluster environment.

        1. Setup Python Virtual Environment.

        ``ivadomed`` requires Python >= 3.6 and <3.9 (If you are using `Compute Canada <https://www.computecanada.ca/>`_, you can load modules (e.g. python 3.9) as `mentioned here <https://intranet.neuro.polymtl.ca/computing-resources/compute-canada#modules>`_ and `also here <https://docs.computecanada.ca/wiki/Utiliser_des_modules/en#Loading_modules_automatically>`_ ). We recommend
        working under a virtual environment, which could be set as follows:

        ::

            virtualenv venv-ivadomed

        2. Activate the new virtual environment (default named `venv-ivadomed`)

        ::

            source venv-ivadomed/bin/activate


        .. warning::
           If the default Python version installed in your system does not fit the version requirements, you might need to specify a version of Python associated with your virtual environment:

           ::

             virtualenv venv-ivadomed --python=python3.6


Step 2: Install Ivadomed with CPU Support
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. tabs::

    .. tab:: Pypi Installation

        Install ``ivadomed`` and its requirements from
        `Pypi <https://pypi.org/project/ivadomed/>`__:

        ::

            pip install --upgrade pip

            pip install ivadomed

    .. tab:: Repo Installation (Advanced or Developer)

        Clone the `ivadomed <https://github.com/ivadomed/ivadomed>`_ repository.
        Install from source

        Bleeding-edge developments are available on the project's master branch
        on Github. Installation procedure is the following at repository root:

        ::

            git clone https://github.com/ivadomed/ivadomed.git

            cd ivadomed

            pip install -e .


(Optional) Step 3: Install IvadoeMed with GPU Support, Install PyTorch 1.5 and TorchVision
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If you have a compatible NVIDIA GPU that supports CUDA11, run the following command:

::

   pip install -r requirements_gpu.txt

According to `nvidia source <https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html>`_, CUDA 11 is compatible with GPUS as far back as `Kepler architecture (GeForce 6xx, 7xx, 8xx series introduced in 2012) <https://en.wikipedia.org/wiki/Kepler_(microarchitecture)>`_ as long as driver is v450+
Cuda Compute Score needs to be > 3.5 as all GPUs listed `here <https://developer.nvidia.com/cuda-gpus>`_.

Please note that this must happens after the previous IvadoMed installation step.

(Optional) Step 4 Install pre-commit hooks for Developers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We use ``pre-commit`` to enforce a limit on file size.
After you've installed ``ivadomed``, install the hooks:

::

    pip install -r requirements_dev.txt
    pre-commit install