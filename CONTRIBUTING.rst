.. _contributing_to_ivadomed:

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

.. _installation_contributor:

Developer ``ivadomed`` installation
++++++++++++++++++++++++++++++++++++++++++++++++++

1. Download the code:

    ::

        cd ivadomed
        git clone https://github.com/ivadomed/ivadomed.git

2. Isolate your work in a virtual environment:

    We recommend developing inside of ``venv`` or ``conda``.
    It makes debugging easier.

    .. tabs::

        .. group-tab:: ``venv``

            Create the environment:

            .. code::

                python -m venv venv


            .. note::

                If you use ``Debian`` or ``Ubuntu``, you may be prompted to
                ``sudo apt install python3-venv`` when creating the virtual environment.
                This is expected, so please follow the instructions provided.

                For other operating systems, ``venv`` will usually be bundled alongside ``python``.

        .. group-tab:: ``conda``

            If you have `conda <https://docs.conda.io/en/latest/miniconda.html>`_ installed, you can
            create a new virtual environment using the provided ``environment.yml``:

                ::

                    conda env create

3. Activate the environment:

    .. tabs::

        .. group-tab:: ``venv``

            .. tabs::

                .. group-tab:: Linux

                    .. code::

                        source venv/bin/activate

                .. group-tab:: Windows

                    .. code::

                        venv\Scripts\activate

                .. group-tab:: macOS

                    .. code::

                        source venv/bin/activate

        .. group-tab:: ``conda``

            ::

                conda activate ivadomed_env


    .. note::

        Every time you come to work on the code you will need to activate the environment in this way.

4. Install the code in "editable" mode:

    ::

        pip install -e .[dev]

    .. note::

        For developing under specific circumstances like exotic GPUs or none at all,
        follow :ref:`the special case install guide <install_special_cases>`, but replace any ``pip install ivadomed`` there with ``pip install -e .[dev]``.




