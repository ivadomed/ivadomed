---
title: 'ivadomed: A Medical Imaging Deep Learning Toolbox'
tags:
  - Deep Learning
  - Medical Imaging
  - Segmentation
  - Open-source
  - Pipeline
authors:
  - name: Charley Gros
    orcid: 0000-0003-4318-0024
    affiliation: 1
  - name: Andreanne Lemay
    orcid: XX
    affiliation: 1
  - name: Olivier Vincent
    orcid: XX
    affiliation: 1
  - name: Lucas Rouhier
    orcid: XX
    affiliation: 1
  - name: Anthime Bucquet
    orcid: XX
    affiliation: 1
  - name: Joseph Paul Cohen
    orcid: XX
    affiliation: "2, 3"
  - name: Julien Cohen-Adad
    orcid: 0000-0003-3662-9532
    affiliation: "1, 4"
affiliations:
 - name: NeuroPoly Lab, Institute of Biomedical Engineering, Polytechnique Montreal, Montreal, Canada
   index: 1
 - name: AIMI, Stanford University, Stanford, CA, USA
   index: 2
 - name: Mila, Quebec AI Institute, Montreal, QC, Canada
   index: 3
 - name:  Functional Neuroimaging Unit, CRIUGM, Université de Montréal, Montreal, QC, Canada
   index: 4
date: 3 November 2020
bibliography: paper.bib

---

# Summary

`ivadomed` is an open-source Python package for designing, end-to-end training, and evaluating deep learning models applied to medical imaging data. The package includes APIs, command-line tools, documentation, and tutorials. `ivadomed` also includes pre-trained models such as spinal tumor segmentation and vertebral labeling. Original features of `ivadomed` include a data loader that can parse image metadata (e.g., acquisition parameters, image contrast, resolution) and subject metadata (e.g., pathology, age, sex) for custom data splitting or extra information during training and evaluation. Any dataset following the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) convention will be compatible with `ivadomed` without the need to manually organize the data, which is typically a tedious task. Beyond the traditional deep learning methods, `ivadomed` features cutting-edge architectures, such as FiLM [@perez2017film] and HeMis [@havaei2016hemis], as well as various uncertainty estimation methods (aleatoric and epistemic), and losses adapted to imbalanced classes and non-binary predictions. Each step is conveniently configurable via a single file. At the same time, the code is highly modular to allow addition/modification of an architecture or pre/post-processing steps. Example applications of `ivadomed` include MRI object detection, segmentation, and labeling of anatomical and pathological structures. Overall, `ivadomed` enables easy and quick exploration of the latest advances in deep learning for medical imaging applications. `ivadomed`'s main project page is available at https://ivadomed.org.

# Statement of need

Deep learning is increasingly used in medical image processing [@kim_deep_2019]. It provides automated solutions to repetitive and/or tedious tasks such as the segmentation of pathological structures. However, medical imaging data present many challenges: datasets are often not publicly-available, ground truth labels are scarce due to the limited availability of expert raters, and needs can be very specific and tailored to particular datasets (e.g., segmentation of spinal tumors on sagittal MRI T2-weighted scans). Thus, offering solution for convenient training models (or fine-tuning of pre-existing models) is needed.

We present `ivadomed`, a deep learning toolbox dedicated to medical data processing. `ivadomed` aims to support the integration of deep learning models into the clinical routine, as well as state-of-the-art academic biomedical research. It features intuitive command-line tools for end-to-end training and evaluation of various deep learning models. The package also includes pre-trained models that can be used to accommodate specific datasets with transfer learning.

Another challenge of medical imaging is the heterogeneity of the data across clinical centers, in terms of image features (e.g., contrast, resolution) and population demographics. This makes it challenging to create models that can generalize well across the many existing datasets. Recent cutting-edge methods address this problem, such as FiLM [@perez2017film] and HeMis [@havaei2016hemis], however they are usually not implemented within a comprehensive framework that enables end-to-end training and experimentation. In addition to providing these advanced architectures, `ivadomed` features multiple uncertainty estimation methods (aleatoric and epistemic), losses adapted to imbalanced classes and non-binary predictions. Each step can be conveniently customized via a single configuration file, and at the same time, the code is highly modular to allow addition/modification of architecture or pre-/post-processing steps.

# Software description

`ivadomed` is based on PyTorch framework [@paszke2017automatic] with GPU acceleration supported by CUDA. It can easily be installed via [PyPI](https://pypi.org/project/ivadomed/) and the whole package is tested with a [continuous integration](https://github.com/ivadomed/ivadomed/actions?query=workflow\%3A\%22Run+tests\%22) framework. The project website, which includes user and API documentation, is available at https://ivadomed.org. The name `ivadomed` is a portmanteau between [*IVADO (The Institute for data valorization)*](https://ivado.ca) and *medical*.

## Loader

XX

## Training

XX

## Evaluation

XX

# Usage

XX # Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

# Acknowledgements

The authors thank Alexandru Jora, Nick Guenther, Christian Perone, Valentine Louis-Lucas, Benoît Sauty-De-Chalon, Alexandru Foias, Marie-Hélène Bourget and Leander Van Eekelen for their useful contributions, and Guillaume Dumas for proof-reading the manuscript. Funded by IVADO, the Canada Research Chair in Quantitative Magnetic Resonance Imaging [950-230815], CIHR [FDN-143263], CFI [32454, 34824], FRQS [28826], NSERC [RGPIN-2019-07244], FRQNT [2020‐RS4‐265502 UNIQUE] and TransMedTech. C.G. has a fellowship from IVADO [EX-2018-4], A.L. has a fellowship from NSERC and FRQNT, O.V. has a fellowship from NSERC, FRQNT and UNIQUE.

# References
