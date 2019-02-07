# IVADO Medical Imaging
This is a repository for the collaboration between MILA and NeuroPoly for the IVADO project on medical imaging.

## Installing
This project requires Python 3.6 and PyTorch >= 1.0, to install all requirements, please use `pip` as described below:

```
~$ git clone https://github.com/neuropoly/ivado-medical-imaging.git
~$ cd ivado-medical-imaging
~$ pip install -e .
```

And all dependencies will be installed into your own system.

## Training
To train the network, use the `ivadomed` command-line tool that will be available on your path after installation, example below:

```
ivadomed config.json
```

The `config.json` is a configuration example.