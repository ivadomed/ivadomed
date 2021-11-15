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

                    .. tab:: Mac/Linux

                        .. code::

                            python3 --version

                    .. tab:: Windows

                        .. code::

                            python --version

                If your system's Python is not 3.6, 3.7, 3.8, or 3.9 (or if you don't have Python 3 installed at all), please `install Python <https://wiki.python.org/moin/BeginnersGuide/Download/>`_ before continuing.

                Once you have a supported version of Python installed, run the following command:


                .. tabs::

                    .. tab:: Mac/Linux

                        .. warning::

                           If you use ``Debian`` or ``Ubuntu``, you may be prompted to install the ``python3-venv`` module when creating the virtual environment. This is expected, so please follow the instructions provided by Python. For other operating systems, ``venv`` will be installed by default.

                        Replacing ``3.X`` with the Python version number that you installed):

                        .. code::

                            python3.X -m venv ivadomed_env

                    .. tab:: Windows

                        .. code::

                            python -m venv ivadomed_env

            2. Activate the new virtual environment (default named ``ivadomed_env``)

                .. tabs::

                    .. tab:: Mac/Linux

                        .. code::

                            source ivadomed_env/bin/activate

                    .. tab:: Windows

                        .. code::

                            cd venv/Scripts/
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


Step 2: Install ``ivadomed``
++++++++++++++++++++++++++++


    .. tabs::

        .. tab:: PyPI Installation

            Install ``ivadomed`` and its requirements from
            `PyPI <https://pypi.org/project/ivadomed/>`__:

            ::

                pip install --upgrade pip

                pip install ivadomed

        .. tab:: Repo Installation (Advanced or Developer)

            Bleeding-edge developments are available on the project's master branch
            on Github. Install ``ivadomed`` from source:

            ::

                git clone https://github.com/ivadomed/ivadomed.git

                cd ivadomed

                pip install -e .


Step 3: Install ``torch`` and ``torchvision`` with CPU or GPU Support
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    .. tabs::

        .. tab:: CPU Support

            If you plan to run ``ivadomed`` on CPU only, install pytorch per instructions provided below for your specific operating system:

            .. tabs::

                .. tab:: Windows/Linux

                    .. code::

                       pip install torch==1.8.0+cpu torchvision==0.9.0+cpu --find-links https://download.pytorch.org/whl/torch_stable.html

                .. tab:: Mac

                    .. code::

                       pip install torch==1.8.0 torchvision==0.9.0 --find-links https://download.pytorch.org/whl/torch_stable.html

                .. tab:: Repo Installation (Advanced or Developer)

                    Run this only if you have already downloaded/cloned the repo with access to the ``requirement.txt`` file, then run the following command while at the repository root level:

                    .. code::

                        pip install -r requirements.txt

        .. tab:: Nvidia GPU Support

            ``ivadomed`` requires CUDA11 to execute properly. If you have a nvidia GPU, try to look up its Cuda Compute Score `here <https://developer.nvidia.com/cuda-gpus>`_, which needs to be > 3.5 to support CUDA11. Then, make sure to upgrade to nvidia driver to be at least v450+ or newer.

            If you have a compatible NVIDIA GPU that supports CUDA11 and with the right driver installed try run the following command relevant to your situation:

            .. tabs::

                .. tab:: All OS

                    .. code::

                       pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 --find-links https://download.pytorch.org/whl/torch_stable.html

                .. tab:: Repo Installation (Advanced or Developer)

                    Run this only if you have already downloaded/cloned the repo with access to the ``requirement_gpu.txt`` file, then run the following command while at the repository root level:
                    .. code::

                       pip install -r requirements_gpu.txt


Developer-only Installation Steps
+++++++++++++++++++++++++++++++++

    The additional steps below are only necessary for contributors to the ``ivadomed`` project.

    The ``pre-commit`` package is used to enforce a size limit on committed files. The ``requirements_dev.txt`` also contain additional dependencies related to documentation building and testing.

    After you've installed ``ivadomed``, install the ``pre-commit`` hooks by running:

    .. code::

        pip install -r requirements_dev.txt
        pre-commit install