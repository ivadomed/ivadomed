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
