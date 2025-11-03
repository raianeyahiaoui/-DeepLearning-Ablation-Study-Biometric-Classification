
# DeepLearning-Ablation-Study-Biometric-Classification

## Project Title: Empirical Analysis of DCNN Performance, Efficiency, and Interpretability for High-Class Biometric Classification

### Contact & Portfolio
| | |
| :--- | :--- |
| **Researcher** | Yahiaoui Raiane |
| **Email** | ikba.king2015@gmail.com |
| **LinkedIn** | [linkedin.com/in/yahiaoui-raiane-253911262](https://www.linkedin.com/in/yahiaoui-raiane-253911262) |

---

### 📖 Project Overview
This repository documents a comprehensive **Ablation Study** focused on optimizing Deep Convolutional Neural Networks (DCNNs) for a challenging 50-class iris recognition task using the UBIRIS.v2 dataset. The goal was to rigorously quantify performance trade-offs across four distinct pipelines. The journey includes validating a baseline, analyzing critical failures (SIFT), implementing robust regularization, and benchmarking a lightweight architecture for edge deployment.

### 🎯 Core Experimental Findings (Referencing Plots in `./images/` Folder)

| Experiment Configuration | Key Result (Val Acc / Loss) | Code Reference | Scientific Diagnosis |
| :--- | :--- | :--- | :--- |
| **1. Peak Performance (ResNet50C)** | **~86.25% / 0.62** | `03/C_Optimal_Performance_ResNet50_C.ipynb` | Achieved the highest performance benchmark through optimal regularization and **pure DCNN feature extraction**. (See: `Optimal_ResNet50C_AccLoss_Benchmark.png`) |
| **2. SIFT Failure Analysis (ResNet50B)** | **~28.00% / 2.83** | `03/A_ResNet50_SIFT_FAILURE_ANALYSIS.ipynb` | **Quantified Failure:** Empirically proved that fusing hand-crafted SIFT features with the DCNN leads to catastrophic under-generalization, validating that the DL model's native features are superior. (See: `images/FAILURE_SIFT_AccLoss_Quantified_ResNet50B.png`) |
| **3. Deployment Viability (MobileNetV2)** | **~80.00% / 1.26** | `03/D_MobileNetV2_Deployment_Viability.ipynb` | **Efficiency Trade-off:** Achieves high, stable accuracy with significantly fewer parameters, validating its optimal suitability for **resource-constrained Embedded Systems**. |
| **4. Debugging Artifact** | **Loss $\sim 5 \times 10^6$** | `03/B_DCNN_Overfitting_and_Regularization.ipynb` | **Numerical Stability Proof:** Evidence of successfully debugging an initial numerical instability issue caused by an unstable learning rate/softmax initialization. (See: `images/DEBUG_Loss_ExtremeNumericalInstability.png`) |

---

### ⚙️ Data Preparation & Feature Exploration

The project began by defining and testing the image pipeline for heterogeneous data:

#### 1. Image Enhancement Pipeline
(Code: `01/A_Initial_Enhancement_Pipeline.py`)
*   **Steps:** Resize $\rightarrow$ Gaussian Blur ($\sigma=5$) $\rightarrow$ Normalization (I/127.5 - 1).
*   **Validation:** This minimal, effective pipeline was proven superior to complex feature engineering. (See: `images/Preprocessing_Color_FullPipeline_Normalized.png` and `images/Preprocessing_NIR_FullPipeline_Normalized.png`).

#### 2. Traditional CV Feature Analysis
(Code: `01/C_SIFT_LBP_Feature_Visualization.ipynb`)
*   **Haar Cascades:** Tested for automated Region-of-Interest (ROI) localization. Analysis revealed limitations and **false positives** on eyelashes/skin in visible-light data, highlighting the need for a DCNN-based segmentation approach (See: `images/Detection_VIS_HaarFailure_FalsePositive.png`).
*   **SIFT & LBP:** Visualized to understand texture encoding (See: `images/Feature_SIFT_LBP_Comparative_Map.png`). This analysis informed the critical decision to abandon the SIFT-fusion pipeline.

### 📈 Model Interpretability and XAI
(Code: `04_Interpretability_Analysis/`)

*   **Activation Visualization:** Feature map grids (See: `images/Interpretability_DeepLayer_ActivationGrid.png`) confirm the DCNN focuses on **high-frequency textural patterns**, validating that the model learns discriminative biometric features.
*   **Error Domain Analysis:** Analysis of misclassified samples (e.g., True C1, Pred C17/C18 - See: `images/ErrorAnalysis_Misclassified_GazeError_Sample.png`) shows primary failures are correlated with **non-frontal gaze** and **heavy occlusion**, defining the limits of the current model's generalization.

### 🚀 Conclusion & Future Work
This project demonstrates that in the fine-grained biometric domain, Transfer Learning with **minimal preprocessing** and **strong regularization (Dropout/Class Weights)** is the optimal strategy. The primary remaining challenge lies in correcting for non-frontal gaze errors and occlusion.

**Future Research Directions (Ph.D. Focus):**
1.  **Metric Learning:** Integrate an ArcFace or Triplet Loss layer to enhance inter-class separation for texture-similar subjects.
2.  **Attention/Normalization:** Implement a dedicated DCNN-based iris segmentation or pose-normalization module to correct for the observed gaze-based errors.
