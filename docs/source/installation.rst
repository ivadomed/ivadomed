.. _installation:
Installation
============

Supported OS
++++++++++++
    
    ``ivadomed`` officially supports GPU acceleration only on ``Linux`` and CPU on ``Linux``, 
    ``Windows`` and ``MacOS``.

.. _installation-step1: 
Step 1: Set up a dedicated virtual environment
++++++++++++++++++++++++++++++++++++++++++++++

    You can set up a virtual environment for ``ivadomed`` using either conda or venv:

    .. tabs::

        .. tab:: Install via ``venv``

            1. Set up Python Venv Virtual Environment.

                ``ivadomed`` requires Python >= 3.7 and <3.10. First, make sure that a
                compatible version of Python 3 is installed on your system by running:

                .. tabs::

                    .. group-tab:: Mac/Linux

                        .. code::

                            python3 --version

                    .. group-tab:: Windows

                        .. code::

                            python --version

                If your system's Python is not 3.7, 3.8, or 3.9 (or if you don't have Python 3 installed at all),
                please `install Python <https://wiki.python.org/moin/BeginnersGuide/Download/>`_ before continuing.

                Once you have a supported version of Python installed, run the following command:

                .. tabs::

                    .. group-tab:: Mac/Linux

                        .. code::

                            python3 -m venv ivadomed_env

                        .. note::

                           If you use ``Debian`` or ``Ubuntu``, you may be prompted to install 
                           the ``python3-venv`` module when creating the virtual environment.
                           This is expected, so please follow the instructions provided by Python.
                           For other operating systems, ``venv`` will be installed by default.

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

                            ivadomed_env\Scripts\activate

        .. tab:: Install via ``conda``

            1. Create a new virtual environment using conda:

                ::

                    conda env create --name ivadomed_env

            2. Activate the created conda environment

                ::

                    conda activate ivadomed_env


        .. tab:: Compute Canada HPC

            There are numerous constraints and limited package availabilities with ComputeCanada cluster environment.

            It is best to attempt ``venv`` based installations and follow up with ComputeCanada technical support as MANY specially compiled packages (e.g. numpy) are exclusively available for Compute Canada HPC environment.

            If you are using `Compute Canada <https://www.computecanada.ca/>`_, you can load modules as `mentioned here <https://intranet.neuro.polymtl.ca/computing-resources/compute-canada#modules>`_ and `also here <https://docs.computecanada.ca/wiki/Utiliser_des_modules/en#Loading_modules_automatically>`_.

.. _installation-step2:
Step 2: Install ``ivadomed``
++++++++++++++++++++++++++++

    .. tabs::
        
        .. tab:: NVIDIA GPU Support

            PyTorch, an integral part of ``ivadomed``, ships
            CUDA 10.2 and CUDA 11.1 runtime by default with its
            respective installation binaries. 

            In case if you're wondering, Ampere-based GPUs 
            (with a `Compute Capability <https://developer.nvidia.com/cuda-gpus>`_
            of 8.x) only work with CUDA>=11.1. Although CUDA 11.1 is
            backward compatible with older hardware, CUDA 10.2 is
            preferred if available.

            Thus, to accelerate ``ivadomed`` with CUDA 10.2 on a Linux system,
            you'd need to have GPUs installed with an `NVIDIA driver version >=440.33 
            <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`_.
            And, for CUDA 11.1 you'd rather need an upgraded NVIDIA driver version >=450.
        
            To verify the NVIDIA driver version, just look in ``/sys`` by 
            executing the command:
                     
            ::

                cat /sys/module/nvidia/version 
            
            and it will return your current driver version.
            
            .. tabs::

                .. tab:: Package Installation (Recommended)

                    To install ``ivadomed`` with CUDA 10.2:

                    ::
                        
                        pip install ivadomed

                    or, with CUDA 11.1:

                    ::

                        pip install ivadomed --extra-index-url https://download.pytorch.org/whl/cu111 

                .. tab:: Source Installation

                    Bleeding-edge developments are available on the master branch of the project
                    on Github. To install ``ivadomed`` from source with CUDA 10.2:
                    
                    ::
            
                        pip install git+https://github.com/ivadomed/ivadomed.git

                    or, with CUDA 11.1:

                    ::

                        pip install git+https://github.com/ivadomed/ivadomed.git \
                        --extra-index-url https://download.pytorch.org/whl/cu111

        .. tab:: CPU Support

            .. tabs:: 
                
                .. tab:: Package Installation (Recommended)

                    .. tabs::

                        .. tab:: Linux

                            ::

                                pip install ivadomed --extra-index-url https://download.pytorch.org/whl/cpu

                        .. tab:: Windows/Mac

                            ::

                                pip install ivadomed
                    
                .. tab:: Source Installation

                    Bleeding-edge developments are available on the project's master branch
                    on Github. To install ``ivadomed`` from source:

                    .. tabs::

                        .. tab:: Linux 

                            ::

                                pip install git+https://github.com/ivadomed/ivadomed.git --extra-index-url https://download.pytorch.org/whl/cpu

                        .. tab:: Windows/Mac 

                            ::

                                pip install git+https://github.com/ivadomed/ivadomed.git