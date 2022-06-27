Contributing to ivadomed
========================

Thank you for your interest in contributing to ivadomed! This project uses the following pages to guide new contributions:

  * The `ivadomed GitHub repository <https://github.com/ivadomed/ivadomed>`_ is where the source code for the project is maintained, and where new contributions are submitted to.
  * The `NeuroPoly Contributing Guidelines <https://intranet.neuro.polymtl.ca/software-development/contributing>`_ provide instructions for development workflows, such as reporting issues or submitting pull requests.
  * The `ivadomed Developer Wiki <https://github.com/ivadomed/ivadomed/wiki>`_ acts as a knowledge base for documenting internal design decisions specific to the ivadomed codebase. It also contains step-by-step walkthroughs for common ivadomed maintainer tasks.

Step 2: Install ``ivadomed``
++++++++++++++++++++++++++++

    .. tabs::
        
        .. tab:: NVIDIA GPU Support

            ``ivadomed`` requires a minimum PyTorch version of either
            1.8.1 or 1.8.2 which supports CUDA 10.2 and CUDA 11.1 builds
            by default. 
            
            Ampere-based GPUs (with a Compute Capability of 8.x) only work
            CUDA>=11.1. Although CUDA 11.1 is backward compatible with older 
            hardware, CUDA 10.2 is preferred if available.

            CUDA 10.2 and CUDA 11.1 require an NVIDIA driver version >=440.33 
            and >=450 respectively as indicated `here <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`__.
            Please make sure to upgrade to the minimum NVIDIA driver version 
            requirements for the respective CUDA builds. To verify the NVIDIA
            driver version, just run the command `nvidia-smi` and you'll find 
            your current driver version on the top.
            
            .. tabs::

                .. tab:: Package Installation (Recommended)

                    To install ``ivadomed`` 

                    .. tabs::

                        .. tab:: Linux

                            with CUDA 10.2:

                            ::
                                
                                pip install ivadomed

                            and, with CUDA 11.1:

                            ::

                                pip install ivadomed --extra-index-url https://download.pytorch.org/whl/cu111 

                        .. tab:: Windows

                            with CUDA 10.2:

                            ::
                                
                                pip install ivadomed --extra-index-url https://download.pytorch.org/whl/cu102

                            and, with CUDA 11.1:

                            ::

                                pip install ivadomed --extra-index-url https://download.pytorch.org/whl/cu111


                .. tab:: Source Installation

                    Bleeding-edge developments are available on the master branch of the project
                    on Github. To install ``ivadomed`` from source

                    .. tabs::

                        .. tab:: Linux

                            with CUDA 10.2:
                            
                            ::
                   
                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e .

                            and, with CUDA 11.1:

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e . --extra-index-url https://download.pytorch.org/whl/cu111

                        .. tab:: Windows 

                            with CUDA 10.2:

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e . --extra-index-url https://download.pytorch.org/whl/cu102

                            and, with CUDA 11.1:

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e . --extra-index-url https://download.pytorch.org/whl/cu111


                .. tab:: Contributor or Developer Installation

                    To contribute to the project, we recommend installing ``ivadomed``
                    from source along with additional dependencies related to building
                    documentation and testing

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

                        .. tab:: Windows

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
                
                .. tab:: Package Installation (Recommended)

                    .. tabs::

                        .. tab:: Linux

                            ::

                                pip install ivadomed --extra-index-url https://download.pytorch.org/whl/cpu

                        .. tab:: Windows 

                            ::

                                pip install ivadomed

                        .. tab:: Mac 

                            ::

                                pip install ivadomed
                    
                .. tab:: Source Installation

                    Bleeding-edge developments are available on the project's master branch
                    on Github. To install ``ivadomed`` from source:

                    .. tabs::

                        .. tab:: Linux 

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu

                        .. tab:: Windows 

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed
                                
                                pip install -e .

                        .. tab:: Mac

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e .

                .. tab:: Contributor or Developer Installation

                    To contribute to the project, we recommend installing ``ivadomed`` from source along with additional dependencies related to building documentation and testing:

                    .. tabs::

                        .. tab:: Linux
                            
                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu

                        .. tab:: Windows 

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e .[dev]

                        .. tab:: Mac

                            ::

                                git clone https://github.com/ivadomed/ivadomed.git

                                cd ivadomed

                                pip install -e .[dev]
