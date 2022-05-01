Installation
============

Supported OS
++++++++++++

    Currently, ``ivadomed`` supports GPU/CPU on ``Linux`` and ``Windows``, and CPU only on ``macOS`` and `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/>`_.

Step 1: Setup dedicated python environment
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    You can setup ``ivadomed`` using either Conda or Venv:

    .. tabs::

        .. tab:: Install via ``venv``

            1. Setup Python Venv Virtual Environment.

                ``ivadomed`` requires Python >= 3.6 and <3.10.

                First, make sure that a compatible version of Python 3 is installed on your system by running:

                .. tabs::

                    .. group-tab:: Mac/Linux

                        .. code::

                            python3 --version

                    .. group-tab:: Windows

                        .. code::

                            python --version

                If your system's Python is not 3.6, 3.7, 3.8, or 3.9 (or if you don't have Python 3 installed at all), please `install Python <https://wiki.python.org/moin/BeginnersGuide/Download/>`_ before continuing.

                Once you have a supported version of Python installed, run the following command:


                .. tabs::

                    .. group-tab:: Mac/Linux

                        .. code::

                            # Replacing ``3.X`` with the Python version number that you installed):
                            python3.X -m venv ivadomed_env

                        .. note::

                           If you use ``Debian`` or ``Ubuntu``, you may be prompted to install the ``python3-venv`` module when creating the virtual environment. This is expected, so please follow the instructions provided by Python. For other operating systems, ``venv`` will be installed by default.

                    .. group-tab:: Windows

                        .. code::

                            python -m venv ivadomed_env

            2. Activate the new virtual environment (default named ``ivadomed_env``)

                .. tabs::

                    .. group-tab:: Mac/Linux

                        .. code::

                            source ivadomed_env/bin/activate

                    .. group-tab:: Windows

                        .. code::

                            cd ivadomed_env/Scripts/
                            activate

        .. tab:: Install via ``conda``

            1. Create new conda environment using ``environment.yml`` file

                ::

                    conda env create --name ivadomed_env

            2. Activate the new conda environment

                ::

                    conda activate ivadomed_env


        .. tab:: Compute Canada HPC

            There are numerous constraints and limited package availabilities with ComputeCanada cluster environment.

            It is best to attempt ``venv`` based installations and follow up with ComputeCanada technical support as MANY specially compiled packages (e.g. numpy) are exclusively available for Compute Canada HPC environment.

            If you are using `Compute Canada <https://www.computecanada.ca/>`_, you can load modules as `mentioned here <https://intranet.neuro.polymtl.ca/computing-resources/compute-canada#modules>`_ and `also here <https://docs.computecanada.ca/wiki/Utiliser_des_modules/en#Loading_modules_automatically>`_.


Step 2: Install ``ivadomed`` with CPU or GPU Support
++++++++++++++++++++++++++++++++++++++++++++++++++++

    .. tabs::

        .. group-tab:: PyPI Installation

            .. tabs::

                .. tab:: NVIDIA GPU Support

                    ``ivadomed`` requires CUDA 11 to execute properly. If you have an NVIDIA GPU, try to look up its Cuda Compute Score `here <https://developer.nvidia.com/cuda-gpus>`__, which needs to be > 3.5 to be compatible with CUDA 11. Also, make sure to upgrade your NVIDIA driver to at least the minimum required version as indicated `here <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`__.

                    To verify the CUDA version, simply run ``nvcc -V`` in both Linux and Windows and the release refers to your current CUDA version.

                    After verifying the CUDA 11 installation, install ``ivadomed`` from `PyPI <https://pypi.org/project/ivadomed/>`__:

                    ::

                        pip install --upgrade pip

                        pip install ivadomed[gpu]


                .. tab:: CPU Support

                    ::

                        pip install --upgrade pip

                        pip install ivadomed[cpu]
                        

        .. group-tab:: Source Installation

            Bleeding-edge developments are available on the project's master branch
            on Github. To install ``ivadomed`` from source:

            .. tabs:: 

                .. tab:: NVIDIA GPU Support

                    ::
                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[gpu]


                .. tab:: CPU Support

                    ::
                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[cpu]

        .. group-tab:: Contributor or Developer Installation

            To install ``ivadomed`` with additional dependencies related to building documentation and testing:

                .. tab:: NVIDIA GPU Support

                    ::
                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev_gpu]


                .. tab:: CPU Support

                    ::
                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev_cpu]
