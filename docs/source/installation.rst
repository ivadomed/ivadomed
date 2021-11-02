Installation
============

Supported OS
++++++++++++

Currently, ``ivadomed`` only supports GPU/CPU on ``Linux`` and ``Windows`` and CPU only on ``macOS`` and `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/>`_.

Step 1: Setup dedicated python environment
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can setup ``ivadomed`` using either Conda or Venv:

.. tabs::

    .. tab:: Install via ``conda``

        This is the easiest way for personal computers.

        1. Create new conda environment using ``environment.yml`` file

        ::

            conda env create --file environment.yml

        2. Activate the new conda environment (default named ``ivadomed_env``)

        ::

            conda activate ivadomed_env

    .. tab:: Install via ``venv``

        1. Setup Python Venv Virtual Environment.

        ``ivadomed`` requires Python >= 3.6 and <3.9 (If you are using `Compute Canada <https://www.computecanada.ca/>`_, you can load modules (e.g. python 3.9) as `mentioned here <https://intranet.neuro.polymtl.ca/computing-resources/compute-canada#modules>`_ and `also here <https://docs.computecanada.ca/wiki/Utiliser_des_modules/en#Loading_modules_automatically>`_ ). We recommend
        working under a virtual environment, which could be set as follows:

        ::

            python -m venv ivadomed_env

        2. Activate the new virtual environment (default named `ivadomed_env`)

        ::

            source ivadomed_env/bin/activate


        .. warning::
           If the default Python version installed in your system does not fit the version requirements, you might need to specify a version of Python associated with your virtual environment:

           ::

             python -m venv ivadomed_env --python=python3.7

    .. tab:: Compute Canada HPC

        There are numerous constraints and limited package availabilities with ComputeCanada cluster environment.

        It is best to attempt ``venv`` based installations and follow up with ComputeCanada technicall support as MANY specially compiled packages (e.g. numpy) are exclusively available for Compute Canada HPC environment.


Step 2: Install ``ivadomed``
++++++++++++++++++++++++++++


.. tabs::

    .. tab:: Pypi Installation

        Install ``ivadomed`` and its requirements from
        `Pypi <https://pypi.org/project/ivadomed/>`__:

        ::

            pip install --upgrade pip

            pip install ivadomed

    .. tab:: Repo Installation (Advanced or Developer)

        Bleeding-edge developments are available on the project's master branch
        on Github. Installation ``ivadomed`` from source:

        ::

            git clone https://github.com/ivadomed/ivadomed.git

            cd ivadomed

            pip install -e .


Step 3: Install ``ivadomed`` with CPU or GPU Support, Install ``torch`` and ``torchvision``
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. tabs::

    .. tab:: CPU Support

        If you plan to run ``ivadomed`` on CPU only, run the following command:
        ::

            pip install -r requirements.txt


    .. tab:: Nvidia GPU Support

        ``ivadomed`` requires CUDA11 to execute properly. If you have a nvidia GPU, try to look up its Cuda Compute Score `here <https://developer.nvidia.com/cuda-gpus>`_, which needs to be > 3.5 to support CUDA11. Then, make sure to upgrade to nvidia driver to be at least v450+ or newer.

        If you have a compatible NVIDIA GPU that supports CUDA11 and with the right driver installed, try run the following command:

        ::

           pip install -r requirements_gpu.txt

        Please note that this must happens after the previous ``ivadomed`` installation step.

Additional Steps Required For ``ivadomed`` Developers: Additional Dependencies and ``pre-commit``
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We use ``pre-commit`` to enforce a limit on file size.
After you've installed ``ivadomed``, install the hooks:

::

    pip install -r requirements_dev.txt
    pre-commit install

Additional Steps Required For ``ivadomed`` Developers on ``macOS``:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The steps below (about updating bashrc) are strongly recommended for MacOS users in the installation process but are optional for Linux/Windows  users.

Open your ``bash``/``zsh`` script file with editor on your Mac.

.. tabs::

    .. tab:: ``zsh`` (most recent macs)

        ::

            vim ~/.zshrc

    .. tab:: ``bash`` (older macs or if manually changed)

        ::

            vim ~/.bashrc


Write in your ``.bashrc``/``.zshrc`` file with following line. This is needed as OS will lock HDF5 generated during training and produce a rather obscure error.

::

    export HDF5_USE_FILE_LOCKING='FALSE'

Save this change and restart your terminal to apply the change.