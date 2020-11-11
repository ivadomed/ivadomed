## v2.5.0 (2020-11-10)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.4.0...v2.5.0)

**BUG**

 - paper.md: Fixed broken link.  [View pull request](https://github.com/ivadomed/ivadomed/pull/517)
 - Change default value of config.json.  [View pull request](https://github.com/ivadomed/ivadomed/pull/514)

**DEPENDENCIES**

 - Requirements.txt: force onnxruntime version.  [View pull request](https://github.com/ivadomed/ivadomed/pull/505)
 - set h5py version in requirements.txt.  [View pull request](https://github.com/ivadomed/ivadomed/pull/500)

**DOCUMENTATION**

 - JOSS submission.  [View pull request](https://github.com/ivadomed/ivadomed/pull/502)

**ENHANCEMENT**

 - Some fixes to logging.  [View pull request](https://github.com/ivadomed/ivadomed/pull/509)

**FEATURE**

 - Training without test set.  [View pull request](https://github.com/ivadomed/ivadomed/pull/498)
 - FiLM for 3D Unet.  [View pull request](https://github.com/ivadomed/ivadomed/pull/491)

**REFACTORING**

 - Refactor utils.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/497)

## v2.4.0 (2020-10-27)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.3.1...v2.4.0)

**BUG**

 - Fix missing version.txt in wheels package.  [View pull request](https://github.com/ivadomed/ivadomed/pull/488)

**DOCUMENTATION**

 - Added reference to arXiv citation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/485)
 - Documenting release workflow.  [View pull request](https://github.com/ivadomed/ivadomed/pull/483)

**ENHANCEMENT**

 - Option to override postprocessing in segment volume.  [View pull request](https://github.com/ivadomed/ivadomed/pull/486)
 - Configuration File Manager.  [View pull request](https://github.com/ivadomed/ivadomed/pull/484)


## v2.3.1 (2020-10-19)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.3.0...v2.3.1)

**BUG**

 - Version format.  [View pull request](https://github.com/ivadomed/ivadomed/pull/481)

## v2.3.0 (2020-10-19) 
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.2.1...v2.3.0)

**BUG**

 - Adapt all metrics to multiclass predictions.  [View pull request](https://github.com/ivadomed/ivadomed/pull/472)
 - fix run_test gpu assignation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/453)

**DOCUMENTATION**

 - Improving documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/477)
 - Tutorial fix.  [View pull request](https://github.com/ivadomed/ivadomed/pull/461)

**ENHANCEMENT**

 - Download data: Add models.  [View pull request](https://github.com/ivadomed/ivadomed/pull/476)
 - Refactoring: Changing print and exit to raise error.  [View pull request](https://github.com/ivadomed/ivadomed/pull/467)
 - Remove "eval" cmd.  [View pull request](https://github.com/ivadomed/ivadomed/pull/465)
 - Custom final activation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/458)
 - Display version.  [View pull request](https://github.com/ivadomed/ivadomed/pull/456)

**FEATURE**

 - Use custom data for film.  [View pull request](https://github.com/ivadomed/ivadomed/pull/460)
 - Uncertainty as post-processing step.  [View pull request](https://github.com/ivadomed/ivadomed/pull/459)

## v2.2.1 (2020-09-22)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.2.0...v2.2.1)

**BUG**

 - Cover image path change on README.  [View pull request](https://github.com/ivadomed/ivadomed/pull/451)

## v2.2.0 (2020-09-22)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.1.0...v2.2.0)

**BUG**

 - Minor fixes prior release.  [View pull request](https://github.com/ivadomed/ivadomed/pull/449)

**DEPENDENCIES**

 - Modify scripts/training_curve.py to avoid tensorflow dependency.  [View pull request](https://github.com/ivadomed/ivadomed/pull/396)

**DOCUMENTATION**

 - Updating documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/425)
 - Tutorial on uncertainty estimation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/399)
 - Tutorial cascaded architecture.  [View pull request](https://github.com/ivadomed/ivadomed/pull/389)

**ENHANCEMENT**

 - Retrain model without resetting weights.  [View pull request](https://github.com/ivadomed/ivadomed/pull/447)
 - Normalized ReLU.  [View pull request](https://github.com/ivadomed/ivadomed/pull/384)
 - Create Ivadomed download function.  [View pull request](https://github.com/ivadomed/ivadomed/pull/379)

**FEATURE**

 - Evenly distribute subjects according to metadata.  [View pull request](https://github.com/ivadomed/ivadomed/pull/423)
 - Resume training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/416)
 - Find optimal threshold with ROC analysis.  [View pull request](https://github.com/ivadomed/ivadomed/pull/383)
 - Generate GIF during training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/374)
 - Add classifier model .  [View pull request](https://github.com/ivadomed/ivadomed/pull/278)

**TESTING**

 - Create coverage and improve testing.  [View pull request](https://github.com/ivadomed/ivadomed/pull/385)

## v2.1.0 (2020-07-21)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.0.2...v2.1.0)

**BUG**

 - Automate training seed.  [View pull request](https://github.com/ivadomed/ivadomed/pull/366)
 - Automate training bug.  [View pull request](https://github.com/ivadomed/ivadomed/pull/363)
 - Apply preprocessing after filter ROI.  [View pull request](https://github.com/ivadomed/ivadomed/pull/342)
 - Fix bug in automate training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/339)
 - Transformations at test time: minor fixes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/335)

**DOCUMENTATION**

 - Documentation: metric more formal defintion.  [View pull request](https://github.com/ivadomed/ivadomed/pull/357)
 - Fix few documentation issues, add content.  [View pull request](https://github.com/ivadomed/ivadomed/pull/341)
 - Soft training: minor fixes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/334)
 - Tutorial 01: One class segmentation 2D Unet.  [View pull request](https://github.com/ivadomed/ivadomed/pull/309)

**ENHANCEMENT**

 - Split dataset with no test center specified.  [View pull request](https://github.com/ivadomed/ivadomed/pull/370)
 - showing time after training (begin/end/duration).  [View pull request](https://github.com/ivadomed/ivadomed/pull/365)
 - Optimize binarization.  [View pull request](https://github.com/ivadomed/ivadomed/pull/364)
 - Automate training improvement.  [View pull request](https://github.com/ivadomed/ivadomed/pull/362)
 - Simplify code when filtering ROI.  [View pull request](https://github.com/ivadomed/ivadomed/pull/361)
 - Scripts: Add entry points, modify doc display, and started to add github action testing.  [View pull request](https://github.com/ivadomed/ivadomed/pull/328)
