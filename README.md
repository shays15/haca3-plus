# 🧠 HACA3+: Harmonizing MR Images Across 100+ Scanners

**Savannah P. Hays et al.**  
*Medical Imaging with Deep Learning (MIDL) 2026 (Validation Track), Taipei, Taiwan — July 2026*

📄 **Paper:** [Harmonizing MR Images Across 100+ Scanners: Multi-site Validation with Traveling Subjects and Real-world Protocols](https://openreview.net/forum?id=TjqfzvRZWg)

## 🚀 Overview

Magnetic resonance (MR) images acquired across different scanners, sites, and protocols exhibit significant variability, which limits the reliability of downstream analysis and machine learning models.

We present HACA3+, an enhanced version of the [HACA3](https://github.com/lianruizuo/haca3) framework for unsupervised MR image harmonization, designed to operate across large-scale, real-world clinical datasets.

HACA3+ introduces:

✅ Improved artifact-aware encoding

✅ Foreground/background-aware attention

✅ Training on 100+ scanners from over 50 sites

## 🔬 Key Contributions

### 1. 🧪 Enhanced Artifact Encoder
- Learns a **continuous artifact severity score**
- Trained using simulated artifacts:
  - noise
  - bias field
  - ghosting
  - anisotropy
- Improves sensitivity to **varying image quality levels**
- Based on prior work:  
  🔗 [Artifact Scoring (Hays et al., MIDL 2025)](https://github.com/shays15/artifact_scoring)

### 2. 🎯 Foreground-Aware Attention
- Replaces slice-wise scalar attention with **foreground/background-aware attention**
- Uses the **union of foreground/background masks**
- Enables:
  - Improved **limited field-of-view (FOV) handling**
  - More accurate **region imputation**

### 3. 🌍 Large-Scale Training
- Multi-contrast structural MR brain images: T1-weighted (T1w), T2-weighted (T2w), FLAIR, and Proton Density (PD)
- Over 100 different scanners represented including five manufacturers (GE, Hitachi, Philips, Siemens, and Toshiba)
