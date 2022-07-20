Installation
============

At your command prompt, run:

::

    pip install ivadomed

This should work with most combinations of OSes and pythons. If not, come ask us for help.

.. TODO: is this note even worth having, given that 99% of people are running Ubuntu which is configured correctly?
.. note::

    If you are installing into your home folder (``pip install --user``), which is the default,
    you need to make sure that ``~/.local/bin/`` (or ``%APPDATA%\Python\PythonXY\Scripts`` on Windows)
    is on your ``$PATH`` for ivadomed's tools to be available. ``pip`` will warn you if it is not.
    It should be pre-configured properly on Ubuntu and most common Linux distros.

    If you are installing into a `venv <https://docs.python.org/3/library/venv.html>`_ or a
    `conda env <https://docs.conda.io/projects/conda/en/latest/commands.html#conda-vs-pip-vs-virtualenv-commands>`_
    then the tools will be available whenever that environment is activated.


.. _install_special_cases:

Special Cases
=============

In rarer cases, you might **instead** use a more specific command,
depending on your hardware and OS.

GPU Support
-----------

Some features in ``ivadomed`` can be accelerated with
the aid of GPU hardware and `PyTorch <https://pytorch.org>`_.
By default, acceleration is available on Linux if
`NVIDIA GPUs <https://developer.nvidia.com/cuda-gpus>`_ are installed in the machine and a compatible
`NVIDIA driver <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`_
is installed in the OS.

After ``pip install ivadomed``, check if acceleration is available by running:

::

  $ python -c 'import torch; print(torch.cuda.is_available())'
  True

If this reports 'False' or something else, you are not configured for GPU acceleration.

.. tabs::

  .. group-tab:: Linux

        To verify your GPU hardware is detected, use ``lspci`` like in:

        ::

            $ lspci -nn -d 10DE::0300
            01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GM200 [GeForce GTX TITAN X] [10de:17c2] (rev a1)
            02:00.0 VGA compatible controller [0300]: NVIDIA Corporation GM200 [GeForce GTX TITAN X] [10de:17c2] (rev a1)

        To verify the NVIDIA driver, examine its version like this example:

        ::

            $ cat /sys/module/nvidia/version
            510.73.05


  .. group-tab:: Windows

        .. TODO: document verifying hardware/driver on Windows

        The default Windows install is CPU-only, because it's still rare to be doing machine-learning
        directly on Windows. But it's possible to use it by asking for a CUDA build:

        .. NOTE: we must periodically update the URL here to use the most recent CUDA,
                 or else pip will start prefering the PyPI build to the pytorch.org build
        .. NOTE: here we recommend CUDA 10.2, because that matches what Linux currently gets from PyPI.
                 If/when torch starts pushing CUDA 11 to PyPI, update this.
        ::

            pip install ivadomed --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu102
  .. group-tab:: macOS

        macOS does not support GPUs. You should treat yourself as being in the `CPU-only Support`_ case.

CUDA 11
~~~~~~~

.. NOTE: If/when torch starts pushing CUDA 11 to PyPI, drop this section.
   (maybe it will need to be reinstated for CUDA 12, but we can cross that bridge whenw e get to it)

If after checking the above, ``torch.cuda.is_available()`` is still not detecting the GPUs, it might be because
they are either too old -- with a `Compute Capability <https://developer.nvidia.com/cuda-gpus>`_ < 3.7 --
or too new -- with a Compute Capability >= 8.0.

If your GPUs are too old, you are out of luck; you should treat yourself as being in the `CPU-only Support`_ case.

But if the GPUs are too new, you can probably get them working by switching
to a CUDA 11.x build. At the moment, this is less well tested than the CUDA 10.x builds,
but it should work:

.. Using --force-reinstall is overwrought, but it's the most reliable one-liner
    to handle switching from one build to another; otherwise pip just says "torch is already installed".
    It would be unnecessary if the user was able to predict which torch they needed,
    from the start, but there's no easy way to instruct them in that besides just trying
    multiple versions, so we're stuck with --force-reinstall.

.. NOTE: we must periodically update the URLs here to cover the most recent CUDA,
            or else pip will start prefering the PyPI build to the pytorch.org build
::

    pip install ivadomed --force-reinstall \
        --extra-index-url https://download.pytorch.org/whl/cu110 \
        --extra-index-url https://download.pytorch.org/whl/cu111 \
        --extra-index-url https://download.pytorch.org/whl/cu112 \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu114 \
        --extra-index-url https://download.pytorch.org/whl/cu115 \
        --extra-index-url https://download.pytorch.org/whl/cu116 \
        --extra-index-url https://download.pytorch.org/whl/cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu118


CPU-only Support
----------------

If you do not own supported GPUs it is a waste of time and (up to 2GB of!) space to install the full GPU build.

.. tabs::

    .. group-tab:: Linux

        ::

            pip install ivadomed --extra-index-url https://download.pytorch.org/whl/cpu
    .. group-tab:: Windows

        Windows' ``torch`` is already CPU-only by default.

        Use the standard installation command.

    .. group-tab:: macOS

        macOS's ``torch`` only supports running in CPU mode.

        Use the standard installation command.



Developer Installation
======================

Interested in contributing to the project? Just head over to the
:ref:`contributing section <contributing_to_ivadomed>` for the guidelines and
contributor specific installation instructions.
