# 🧠 HACA3+: Harmonizing MR Images Across 100+ Scanners

MIDL 2026 (Validation Track)
Savannah P. Hays et al.
Harmonizing MR Images Across 100+ Scanners: Multi-site Validation with Traveling Subjects and Real-world Protocols
Link: https://openreview.net/forum?id=TjqfzvRZWg&referrer=%5Bthe%20profile%20of%20Savannah%20P%20Hays%5D(%2Fprofile%3Fid%3D~Savannah_P_Hays1)

# 🚀 Overview

Magnetic Resonance (MR) images acquired across different scanners, sites, and protocols exhibit significant variability, which limits the reliability of downstream analysis and machine learning models.

We present HACA3+, an enhanced version of the HACA3 framework for unsupervised MR image harmonization, designed to operate across large-scale, real-world clinical datasets.

HACA3+ introduces:

✅ Improved artifact-aware encoding
✅ Foreground/background-aware attention
✅ Training on 100+ scanners from 64 sites

Our work focuses on robust validation at scale, rather than proposing a new architecture.

# 🔬 Key Contributions
1. Enhanced Artifact Encoder
Learns a continuous artifact severity score
Trained using simulated artifacts (noise, bias field, ghosting, anisotropy)
Improves sensitivity to varying image quality levels
2. Spatially-Aware Attention
Replaces slice-wise scalar attention with pixel-wise attention
Uses union of foreground/background masks
Enables:
Better limited field-of-view (FOV) handling
Improved region imputation
3. Large-Scale Training
996 subjects
64 sites
132 scanners
Covers:
T1-weighted (T1w)
T2-weighted (T2w)
FLAIR
Proton Density (PD)
