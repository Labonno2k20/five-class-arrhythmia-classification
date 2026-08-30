# Five-Class Arrhythmia Classification Using ECG Signals

A deep learning-based framework for five-class cardiac arrhythmia classification from ECG signals using signal preprocessing, SWT-Hilbert Transform-based feature extraction, deep learning, Ant Colony Optimization, and Explainable AI.

## Overview

This project presents an automated ECG heartbeat classification framework designed to classify cardiac beats into five AAMI categories using the MIT-BIH Arrhythmia Database.

The framework combines ECG signal preprocessing, time-frequency feature extraction, rhythm-related information, deep learning classification, hyperparameter optimization, and Explainable AI (XAI).

### Classified Arrhythmia Classes

| Class | Description              |
| ----- | ------------------------ |
| **N** | Normal                   |
| **S** | Supraventricular ectopic |
| **V** | Ventricular ectopic      |
| **F** | Fusion                   |
| **Q** | Unknown                  |

## Methodology

The complete workflow consists of:

```text
MIT-BIH Arrhythmia Database
            ↓
ECG Signal Preprocessing
            ↓
Baseline Wander Removal
            ↓
DWT-Based Denoising
            ↓
R-Peak Detection & Heartbeat Segmentation
            ↓
SWT-Hilbert Transform
            ↓
Instantaneous Amplitude, Phase & Frequency
            ↓
R-R Interval Feature
            ↓
SMOTE-Based Class Balancing
            ↓
Deep Learning Classification
        ↙           ↘
 Attention          FCN
   Bi-LSTM
        ↘           ↙
 Ant Colony Optimization
            ↓
 Five-Class Classification
            ↓
      Explainable AI
           (SHAP)
```

## ECG Preprocessing

The ECG signals were processed to reduce noise and obtain suitable heartbeat segments for classification.

The preprocessing pipeline includes:

* Baseline wander removal
* DWT-based signal denoising
* R-peak detection
* Heartbeat segmentation
* Z-score normalization

Each heartbeat was segmented using a window of **100 samples before** and **180 samples after** the R-peak, resulting in **280 samples per heartbeat**.

The ECG recordings were sampled at **360 Hz**.

## Feature Extraction

The **SWT-Hilbert Transform (SWT-HT)** was used to extract time-frequency characteristics from the ECG signals.

The extracted features include:

* Instantaneous Amplitude (IA)
* Instantaneous Phase (IP)
* Instantaneous Frequency (IF)

An additional **R-R interval feature** was incorporated to provide rhythm-related information that may not be sufficiently represented by morphological features alone.

The resulting feature representation contains **37 features** per heartbeat.

## Data Balancing

The dataset contains a substantial imbalance between normal and arrhythmic heartbeat classes.

**SMOTE (Synthetic Minority Over-sampling Technique)** was applied to address class imbalance before model training.

## Deep Learning Models

Two deep learning architectures were developed and optimized:

### Attention-Based Bi-LSTM

A Bidirectional Long Short-Term Memory (Bi-LSTM) architecture with an attention mechanism was developed to capture temporal dependencies within the ECG feature sequences.

### Fully Convolutional Network

A Fully Convolutional Network (FCN) was developed to learn discriminative patterns from the extracted ECG features using one-dimensional convolutional operations.

## Ant Colony Optimization

**Ant Colony Optimization (ACO)** was used for hyperparameter optimization of the deep learning models.

The resulting optimized models are referred to as:

* **ACO-Bi-LSTM**
* **ACO-FCN**

## Explainable AI

**SHAP (SHapley Additive exPlanations)** was used to investigate the contribution of the extracted features to the model predictions.

The explainability analysis provides insight into:

* Feature importance
* Feature contributions to predictions
* Model decision patterns
* Differences in feature importance across arrhythmia classes

## Results

The optimized models achieved the following overall performance:

| Model           |   Accuracy |  Precision |     Recall |   F1-Score | Specificity |
| --------------- | ---------: | ---------: | ---------: | ---------: | ----------: |
| **ACO-FCN**     | **99.77%** | **99.43%** | **99.42%** | **99.42%** |  **99.85%** |
| **ACO-Bi-LSTM** | **99.40%** | **98.52%** | **98.51%** | **98.50%** |  **99.63%** |

## Results and Visualizations

The `results/figures/` directory contains the major visualizations generated during the study.

### Preprocessing

![Preprocessing](results/figures/preprocessing.png)

### R-Peak Segmentation

![R-Peak Segmentation](results/figures/R_peak_segmentation.png)

### Z-Score Normalization

![Z-Score Normalization](results/figures/z_score_normalization.png)

### ACO-FCN Training Curves

![ACO-FCN Curves](results/figures/ACO_FCN_Curves.png)

### ACO-FCN Confusion Matrix

![ACO-FCN Confusion Matrix](results/figures/ACO_FCN_Confusion_Matrix_BW.png)

### ACO-Bi-LSTM Training Curves

![ACO-Bi-LSTM Curves](results/figures/ACO_BiLSTM_curves_BW.png)

### ACO-Bi-LSTM Confusion Matrix

![ACO-Bi-LSTM Confusion Matrix](results/figures/ACO_BiLSTM_Confusion_Matrix_BW.png)

### SHAP Analysis — ACO-FCN

![ACO-FCN SHAP](results/figures/ACO_FCN_Grayscale_SHAP_Final.png)

### SHAP Analysis — Attention ACO-Bi-LSTM

![Attention ACO-Bi-LSTM SHAP](results/figures/Attention_ACO_BiLSTM_Grayscale_SHAP_Final.png)

## Repository Structure

```text
five-class-arrhythmia-classification/
│
├── arrhythmia_classification.ipynb
│
├── results/
│   └── figures/
│       ├── ACO_BiLSTM_Confusion_Matrix_BW.png
│       ├── ACO_BiLSTM_curves_BW.png
│       ├── ACO_FCN_Confusion_Matrix_BW.png
│       ├── ACO_FCN_Curves.png
│       ├── ACO_FCN_Grayscale_SHAP_Final.png
│       ├── Attention_ACO_BiLSTM_Grayscale_SHAP_Final.png
│       ├── preprocessing.png
│       ├── R_peak_segmentation.png
│       └── z_score_normalization.png
│
├── .gitignore
│
└── README.md
```

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* PyWavelets
* SHAP
* Matplotlib
* Seaborn

## Dataset

This project uses the **MIT-BIH Arrhythmia Database** from PhysioNet.

The original dataset is not included in this repository.

## Notebook

The complete implementation is available in:

`arrhythmia_classification.ipynb`

The notebook contains the ECG preprocessing, feature extraction, data balancing, model development, optimization, evaluation, and SHAP-based explainability workflow.

## Author

**Labonno**

Biomedical Engineering

2026
