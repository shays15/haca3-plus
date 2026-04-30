# 🧠 HACA3+: Harmonizing MR Images Across 100+ Scanners

**This page is currently under construction. Code is available. Pre-trained weights and singularity container coming May 2026!**

### <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30">Recent Updates
- 📄 [Harmonizing MR Images Across 100+ Scanners: Multi-site Validation with Traveling Subjects and Real-world Protocols](https://openreview.net/forum?id=TjqfzvRZWg) accepted for poster presentation at *Medical Imaging with Deep Learning (MIDL) 2026 (Validation Track), Taipei, Taiwan — July 8-10, 2026*.
    - Citation:    
        ```bibtex
        @inproceedings{hays2026midl,
        title = {{Harmonizing MR Images Across 100+ Scanners: Multi-site
        Validation with Traveling Subjects and Real-world Protocols}},
        booktitle = {Proceedings of Medical Imaging with Deep Learning~(MIDL
        2026), Taipei, Taiwan, July 8 -- 10, 2026},
        pages = {},
        year = {2026},
        author = {S.~P. Hays and L. Zuo and M.~F.~A. Chaudhary and K.~M.
        Bartz and S.~W. Remedios J. Zhang and M. Bilgel and E.~M. Mowry and
        S.~D. Newsome and B.~E. Dewey and J.~L. Prince and A. Carsss}
        }
        ```

## 🚀 Overview

Magnetic resonance (MR) images acquired across different scanners, sites, and protocols exhibit significant variability, which limits the reliability of downstream analysis and machine learning models.

We present HACA3+, an enhanced version of the [HACA3](https://github.com/lianruizuo/haca3) framework for unsupervised MR image harmonization, designed to operate across large-scale, real-world clinical datasets.

HACA3+ introduces:

✅ Improved artifact-aware encoding

✅ Foreground/background-aware attention

✅ Training on 100+ scanners from over 50 sites

## 🔬 Key Contributions

### 🧪 Enhanced Artifact Encoder
- Learns a **continuous artifact severity score**
- Trained using simulated artifacts:
  - noise
  - bias field
  - ghosting
  - anisotropy
- Improves sensitivity to **varying image quality levels**
- Based on prior work:  
  🔗 [Artifact Scoring (Hays et al., MIDL 2025)](https://github.com/shays15/artifact_scoring)

### 🎯 Foreground-Aware Attention
- Replaces slice-wise scalar attention with **foreground/background-aware attention**
- Uses the **union of foreground/background masks**
- Enables:
  - Improved **limited field-of-view (FOV) handling**
  - More accurate **region imputation**

### 🌍 Large-Scale Training
- Multi-contrast structural MR brain images: T1-weighted (T1w), T2-weighted (T2w), FLAIR, and Proton Density (PD)
- Over 100 different scanners represented including five manufacturers (GE, Hitachi, Philips, Siemens, and Toshiba)


## Prerequisites
Standard neuroimage preprocessing steps are needed before running HACA3. These preprocessing steps include:
- Inhomogeneity correction
- Super-resolution for 2D acquired scans. This step is optional, but recommended for optimal performance. See [ECLARE](https://github.com/sremedios/eclare) for more details.
- Registration to MNI space (1mm isotropic resolution). HACA3 assumes a spatial dimension of 192x224x192.

## Installation and pretrained weights

### Option 1 (recommended): Run HACA3 through singularity image
In general, no installation of HACA3 is required with this option. 
Singularity image of HACA3 model can be directly downloaded [**here**](TODO).


### Option 2: Install from source using `pip`
1. Clone the repository:
    ```bash
    git clone https://github.com/shays15/haca3-plus.git 
    ```
2. Navigate to the directory:
    ```bash
    cd haca3
    ```
3. Install dependencies:
    ```bash
    pip install . 
    ```
Package requirements are automatically handled. To see a list of requirements, see `setup.py` L50-60. 
This installs the `haca3` package and creates two CLI aliases `haca3-train` and `haca3-test`.


### Pretrained weights
Pretrained weights of HACA3 can be downloaded [**here**](TODO). 
HACA3 uses a 3D convolutional network to combine multi-orientation 2D slices into a single 3D volume. 
Pretrained fusion model can be downloaded [**here**](TODO).

## Usage: Inference

### Option 1 (recommended): Run HACA3 through singularity image
   ```bash
   singularity exec --nv -e haca3.sif haca3-test \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-1] \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-2, IF THERE ARE MULTIPLE SOURCE IMAGES] \
   --target-image [TARGET-IMAGE] \
   --harmonization-model [PRETRAINED-HACA3-MODEL] \
   --fusion-model [PRETRAINED-FUSION-MODEL] \
   --out-path [PATH-TO-HARMONIZED-IMAGE] \
   --intermediate-out-dir [DIRECTORY SAVES INTERMEDIATE RESULTS] 
   ```

- ***Example:***
    Suppose the task is to harmonize MR images from `Site A` to match the contrast of a pre-selected T1w image of 
    `Site B`. As a source site, `Site A` has T1w, T2w, and FLAIR images. The files are saved like this:
    ```
    ├──data_directory
        ├──site_A_t1w.nii.gz
        ├──site_A_t2w.nii.gz
        ├──site_A_flair.nii.gz
        └──site_B_t1w.nii.gz
    ```
    You can always retrain HACA3 using your own datasets. In this example, we choose to use the pretrained HACA3 weights 
    `harmonization.pt` and fusion model weights `fusion.pt`. The singularity command to run HACA3 is:
    ```bash
       singularity exec --nv -e haca3.sif haca3-test \
       --in-path data_directory/site_A_t1w.nii.gz \
       --in-path data_directory/site_A_t2w.nii.gz \
       --in-path data_directory/site_A_flair.nii.gz \
       --target-image data_directory/site_B_t1w.nii.gz \
       --harmonization-model harmonization.pt \
       --fusion-model fusion.pt \
       --out-path output_directory/site_A_harmonized_to_site_B_t1w.nii.gz \
       --intermediate-out-dir output_directory
    ```
    The harmonized image and intermediate results will be saved at `output_directory`.


### Option 2: Run HACA3 from source after installation
   ```bash
   haca3-test \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-1] \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-2, IF THERE ARE MULTIPLE SOURCE IMAGES] \
   --target-image [TARGET-IMAGE] \
   --harmonization-model [PRETRAINED-HACA3-MODEL] \
   --fusion-model [PRETRAINED-FUSION-MODEL] \
   --out-path [PATH-TO-HARMONIZED-IMAGE] \
   --intermediate-out-dir [DIRECTORY-THAT-SAVES-INTERMEDIATE-RESULTS] 
   ```


### All options for inference
- ```--in-path```: file path to input source image. Multiple ```--in-path``` may be provided if there are multiple 
source images. See the above example for more details.
- ```--target-image```: file path to target image. HACA3 will match the contrast of source images to this target image.
- ```--target-theta```: In HACA3, ```theta``` 
is a two-dimensional representation of image contrast. Target image contrast can be directly specified by providing 
a ```theta``` value, e.g., ```--target-theta 0.5 0.5```. Note: either ```--target-image``` or ```--target-image``` must 
be provided during inference. If both are provided, only ```--target-theta``` will be used.
- ```--norm-val```: normalization value. 
- ```--out-path```: file path to harmonized image. 
- ```--harmonization-model```: pretrained HACA3 weights. Pretrained model weights on can 
be downloaded [here](TODO).
- ```--fusion-model```: pretrained fusion model weights. HACA3 uses a 3D convolutional network to combine multi-orientation
2D slices into a single 3D volume. Pretrained fusion model can be downloaded [here](TODO).
- ```--save-intermediate```: if specified, intermediate results will be saved. Default: ```False```. Action: ```store_true```.
- ```--intermediate-out-dir```: directory to save intermediate results.
- ```--gpu-id```: integer number specifies which GPU to run HACA3.
- ```--num-batches```: During inference, HACA3 takes entire 3D MRI volumes as input. This may cause a considerable amount 
GPU memory. For reduced GPU memory consumption, source images maybe divided into smaller batches. 
However, this may slightly increase the inference time.

## Go further with harmonization 
- ***Application #1: Identifying optimal operating contrast.*** With the ability of synthesizing arbitrary 
contrasts of the same underlying anatomy, we use harmonization to identify the optimal operating contrast (OOC) of various
downstream tasks, e.g., different segmentation algorithms. 
  - Publications:   
    [Hays et al. Evaluating the Impact of MR Image Contrast on Whole Brain Segmentation. SPIE 2022.](https://drive.google.com/file/d/1ZxLqJCFORPqhwZCQVM_7r7TwZcn5bbzy/view)   
    [Hays et al. Exploring the Optimal Operating MR Contrast for Brain Ventricle Parcellation. MIDL 2023.](https://openreview.net/pdf?id=3ndjE9eawkr)   
    [Hays et al. Optimal operating MR contrast for brain ventricle parcellation. ISBI 2023.](https://arxiv.org/pdf/2304.02056)   
  
- ***Application #2: Automatic quality assurance.*** Since HACA3 has the ability of identifying images with high artifact
levels, we use the HACA3 artifact encoder to do automatic quality assurance. 
  - Publication:    
    [Zuo et al. A latent space for unsupervised MR image quality control via artifact assessment. SPIE 2023.](https://arxiv.org/pdf/2302.00528)

- ***Application #3: Consistent longitudinal analysis.*** We have identified that inconsistent acquisition can cause 
significant issues in longitudinal volumetric analysis, and harmonization is a solution to alleviate this issue of inconsistency.
  - Publication:   
    [Zuo et al. Inconsistent MR Acquisition in Longitudinal Volumetric Analysis: Impacts and Solutions. CMSC 2023.](https://cmsc.confex.com/cmsc/2023/meetingapp.cgi/Paper/8967) 
  - Video presentation on [YouTube](https://www.youtube.com/watch?v=TpdB55wxgs4&t=2s)

- ***Application #4: Quantifying scanner differences from images.*** In many cases, scanner and acquisition information is 
not immediately available from NIFTI files. The contrast encoder in HACA3 and our previous harmonization model 
[CALAMITI](https://www.sciencedirect.com/science/article/pii/S1053811921008429) provides a way to capture these acquisition differences 
from MR images themselves. This information can be used to inform downstream tasks about the level of data heterogeneity. 
  - Publication:   
    [Hays et al. Quantifying Contrast Differences Among Magnetic Resonance Images Used in Clinical Studies. CMSC 2023.](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=pMxz1VYAAAAJ&citation_for_view=pMxz1VYAAAAJ:qjMakFHDy7sC)


## Acknowledgements
This research is partially supported by the Johns Hopkins University Percy Pierre Fellowship~(Hays) and the National Science Foundation Graduate Research Fellowship under Grant No. DGE-2139757~(Hays).
Development is partially supported by FG-2008-36966~(Dewey), CDMRP W81XWH2010912~(Prince), NIH R01EB036013~(Prince), NIH R01 CA253923~(Landman), NIH R01 CA275015~(Landman), the National MS Society grant RG-1507-05243~(Pham) and Patient-Centered Outcomes Research Institute~(PCORI) grant MS-1610-37115~(Newsome and Mowry).
The statements in this publication are solely the responsibility of the authors and do not necessarily represent the views of the Patient-Centered Outcomes Research Institute~(PCORI), its Board of Governors or Methodology Committee.

This research was supported in part by the Intramural Research Program of the National Institutes of Health~(NIH). The contributions of the NIH author(s) were made as part of their official duties as NIH federal employees, are in compliance with agency policy requirements, and are considered Works of the United States Government. However, the findings and conclusions presented in this paper are those of the author(s) and do not necessarily reflect the views of the NIH or the U.S. Department of Health and Human Services.

Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

Data were provided [in part] by OASIS-3: Longitudinal Multimodal Neuroimaging: Principal Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P30 AG066444, P50 AG00561, P30 NS09857781, P01 AG026276, P01 AG003991, R01 AG043434, UL1 TR000448, R01 EB009352.
