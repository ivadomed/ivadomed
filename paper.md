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

`ivadomed` is an open-source Python package for designing, end-to-end training, and evaluating deep learning models applied to medical imaging data. The package includes APIs, command-line tools, documentation, and tutorials. `ivadomed` also includes pre-trained models such as spinal tumor segmentation and vertebral labeling. Original features of `ivadomed` include a data loader that can parse image and subject metadata for custom data splitting or extra information during training and evaluation. Any dataset following the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) convention will be compatible with `ivadomed`. Beyond the traditional deep learning methods, `ivadomed` features cutting-edge architectures, such as FiLM [@perez2017film] and HeMis [@havaei2016hemis], as well as various uncertainty estimation methods (aleatoric and epistemic), and losses adapted to imbalanced classes and non-binary predictions. Example applications of `ivadomed` include MRI object detection, segmentation, and labeling of anatomical and pathological structures. `ivadomed`'s main project page is available at https://ivadomed.org.

# Statement of need

Deep learning is increasingly used in medical image processing [@kim_deep_2019]. It provides automated solutions to repetitive and/or tedious tasks such as the segmentation of pathological structures. However, medical imaging data present many challenges: datasets are often not publicly-available, ground truth labels are scarce due to the limited availability of expert raters, and needs can be very specific and tailored to particular datasets and tasks. There already exists a variety of deep learning framework for medical imaging, but [each have their pros & cons](https://ivadomed.org/en/latest/purpose.html#comparison-with-other-projects). `ivadomed` notably address unmet needs in terms of data management (BIDS), readily-available uncertainty outputs, missing channels (in case of multi-channel training) and extensive framework for model comparison, to only name few of the original features.

Another challenge of medical imaging is the heterogeneity of the data across clinical centers, in terms of image features (e.g., contrast, resolution) and population demographics. This makes it challenging to create models that can generalize well across the many existing datasets. Recent cutting-edge methods address this problem, such as FiLM [@perez2017film] and HeMis [@havaei2016hemis], however they are usually not implemented within a comprehensive framework that enables end-to-end training and experimentation. In addition to providing these advanced architectures, `ivadomed` features multiple uncertainty estimation methods (aleatoric and epistemic), losses adapted to imbalanced classes and non-binary predictions. Each step can be conveniently customized via a single configuration file, and at the same time, the code is highly modular to allow addition/modification of architecture or pre-/post-processing steps.

![Overview of `ivadomed` main features.\label{fig:overview}](https://raw.githubusercontent.com/ivadomed/doc-figures/main/overview.png)

## Loader

In machine learning, lots of time is spent curating the data (renaming, filtering per feature) before entering the training pipeline [@Willemink2020-au]. `ivadomed` features an advanced data loader compatible with a standardized data structure in neuroimaging: the Brain Imaging Data Structure (BIDS) [@bids_2016]. Thus, any dataset following the BIDS convention can readily be used by `ivadomed`. BIDS convention is originally designed around MRI data and accepts NIfTI file formats, but the BIDS community is actively expanding its specifications to other modalities (CT, MEG/EEG, microscopy), which `ivadomed` will then be able to accommodate.

One benefit of the BIDS convention is that each image file is associated with a JSON file containing metadata. `ivadomed`'s loader can parse image metadata (e.g., acquisition parameters, image contrast, resolution) and subject metadata (e.g., pathology, age, sex) for custom data splitting or extra information during training and evaluation. It is possible to modulate specific layers of a convolutional neural network using metadata information (e.g., image contrast, data center, disease phenotype), to tailor it towards a particular data domain or to enable experiments with architectures such as FiLM [@perez2017film] (which is implemented in `ivadomed`). Metadata could also be useful to mitigate class imbalance via data balancing techniques.

`ivadomed`'s data loader can accommodate 2D/3D images, multiple input channels as well as multiple prediction classes. Images can be loaded as a volume, slice-wise, or patch-wise. Data can be saved on the RAM or used "on the fly" via HDF5 format. Cropping, resampling, normalization, and histogram clipping and equalization can be applied during the loading as a pre-processing step. `ivadomed` can deal with missing modalities [@havaei2016hemis] by resorting to curriculum learning to train the model.

## Training

`ivadomed` includes all the necessary components for training segmentation models from start to finish, including data augmentation transformations and transfer learning. Available architectures include: 2D U-Net [@Ronneberger2015unet], 3D U-Net [@isensee2017brain], ResNet [@he2016resnet], DenseNet [@Huang2017densenet], Count-ception [@Cohen2017countception], and HeMIS U-Net. These models can easily be enriched via attention blocks [@oktay2018attention] or FiLM layers (which modulate U-Net features using metadata). To facilitate the training process, `ivadomed` offers multiple loss functions such as the Dice coefficient [@milletari2016v], cross-entropy, and L2 norm, including some adapted to medical imaging challenges, such as the adaptive wing loss [@wang_adaptive_2019] for soft labels and the focal Dice loss [@wong20183d] for class imbalance. To partly address the problem of small datasets, mixup [@zhang2017mixup] has been modified to handle segmentation tasks. To mitigate class imbalance, `ivadomed` supports cascaded architectures. With a single inference, it is possible to narrow down the region of interest via object detection, and then segment a specific structure, as shown in \autoref{fig:lemay2020}. In case of interruption during training, all parameters are saved after each epoch so that training can be resumed at any time.

It can be overwhelming to get started and choose across all the available models, losses, and parameters. `ivadomed`'s repository includes the script `ivadomed_automate_training` to configure and launch multiple trainings across GPUs.

## Evaluation

A model can be thoroughly evaluated on the testing set by computing various popular metrics for segmentation, classification, and regression tasks. Slice-wise or patch-wise predictions are reconstructed in the initial space for evaluation and output visualization. `ivadomed` can produce aleatoric [@wang_aleatoric_2019] and/or epistemic [@nair_exploring_2018] uncertainty, voxel-wise and/or object-wise [@roy_quicknat_2018], using multiple available measures (e.g., entropy, coefficient of variation). Results are reported in a CSV file. The evaluation framework can be further customized with post-processing (e.g., fill holes, remove small objects, thresholding using uncertainty). It is also possible to compute metrics for specific object sizes (e.g., small vs. large lesions). `ivadomed` has a module to find the optimal threshold value on the output soft prediction, via a grid-search finding applied to evaluation metrics or ROC curve.

Multiple visualization tools are included in `ivadomed` to support the design and optimization of tailored training models: GIF animations across training epochs, visual quality control of data augmentation, training curve plots, integration of the TensorBoard module, and output images with true/false positive labels.

# Usage

Past or ongoing research projects using `ivadomed` are listed [here](https://github.com/ivadomed/ivadomed/docs/source/use_cases.rst). For example, `ivadomed` has been used to create the framework depicted in \autoref{fig:lemay2020}, specifically designed for spinal cord tumor segmentation on MRI data [@lemay_fully_2020].

![Fully automatic spinal cord tumor segmentation framework..\label{fig:lemay2020}](https://raw.githubusercontent.com/ivadomed/doc-figures/main/use_cases/lemay_2020.png)

# Acknowledgements

The authors thank Alexandru Jora, Nick Guenther, Christian Perone, Valentine Louis-Lucas, Benoît Sauty-De-Chalon, Alexandru Foias, Marie-Hélène Bourget and Leander Van Eekelen for their useful contributions, and Guillaume Dumas for proof-reading the manuscript. Funded by IVADO, the Canada Research Chair in Quantitative Magnetic Resonance Imaging [950-230815], CIHR [FDN-143263], CFI [32454, 34824], FRQS [28826], NSERC [RGPIN-2019-07244], FRQNT [2020‐RS4‐265502 UNIQUE] and TransMedTech. C.G. has a fellowship from IVADO [EX-2018-4], A.L. has a fellowship from NSERC and FRQNT, O.V. has a fellowship from NSERC, FRQNT and UNIQUE.

# References
