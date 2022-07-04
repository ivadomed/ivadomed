Contributing to ivadomed
========================

General Guidelines
++++++++++++++++++

Thank you for your interest in contributing to ivadomed! This project uses the following pages to guide new contributions:

  * The `ivadomed GitHub repository <https://github.com/ivadomed/ivadomed>`_
    is where the source code for the project is maintained, and where new
    contributions are submitted to. We welcome any type of contribution
    and recommend setting up ``ivadomed`` by following the Contributor
    or Developer installation as instructed below before proceeding
    towards any contribution.

  * The `NeuroPoly Contributing Guidelines <https://intranet.neuro.polymtl.ca/software-development/contributing>`_ 
    provide instructions for development workflows, such as reporting issues or submitting pull requests.

  * The `ivadomed Developer Wiki <https://github.com/ivadomed/ivadomed/wiki>`_
    acts as a knowledge base for documenting internal design decisions specific
    to the ivadomed codebase. It also contains step-by-step walkthroughs for
    common ivadomed maintainer tasks.

Contributor or Developer ``ivadomed`` installation 
++++++++++++++++++++++++++++++++++++++++++++++++++

    .. tabs::
        
        .. tab:: NVIDIA GPU Support

            PyTorch, integral part of ``ivadomed``, ships 
            CUDA 10.2 and CUDA 11.1 runtime by default with its
            respective installation binaries.

            In case if you're wondering, Ampere-based GPUs
            (with a `Compute Capability <https://developer.nvidia.com/cuda-gpus>`_
            of 8.x) only work with CUDA>=11.1. Although CUDA 11.1 is
            backward compatible with older hardware, CUDA 10.2 is
            preferred if available.

            Thus, to accelerate ``ivadomed`` with CUDA 10.2 on a Linux system, you'd
            need to have GPUs installed with an `NVIDIA driver version >=440.33 
            <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`_.
            And, for CUDA 11.1 you'd rather need an upgraded NVIDIA driver version >=450.
            
            To verify the NVIDIA driver version, look in ``/sys`` by executing the
            command:

            ::

                cat /sys/module/nvidia/version
                
            and it will return your current driver version.
            
            Before proceeding with the installation, we suggest you to set up a virtual environment
            by following the instructions as specified in the :ref:`step 1 of the installation <installation_step1>`.

            Once you've set up a virtual environment and activated it, we recommend installing 
            ``ivadomed`` from source along with some additional dependencies related to building
            documentation, linting and testing:

            .. tabs::

                .. tab:: Linux

                    with CUDA 10.2:

                    ::

                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev]

                    with CUDA 11.1:

                    ::

                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu111

                .. tab:: Mac/Windows

                    with CUDA 10.2:

                    ::

                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu102

                        
                    with CUDA 11.1:

                    ::

                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu111


        .. tab:: CPU Support

            .. tabs::

                .. tab:: Linux
                    
                    ::

                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu

                .. tab:: Mac/Windows

                    ::

                        git clone https://github.com/ivadomed/ivadomed.git

                        cd ivadomed

                        pip install -e .[dev]
