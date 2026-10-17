# Parkinson's Disease Detection from Gait

Binary classification of Parkinson's disease (PD) vs. healthy controls from gait signals, solved in two ways:

| Task | Data | Approach | Best result |
|------|------|----------|-------------|
| **1. Feature-based ML** | Vertical ground reaction force (foot pressure sensors) | Interpretable biomechanical features → SVM (RBF) / Random Forest | **ROC-AUC 0.958 ± 0.044**, F1 0.917 |
| **2. Deep learning on time series** | Wearable accelerometers (5 IMU sensors) | Sliding windows → CNN-BiLSTM → patient-level aggregation | **Patient-level AUC 0.889**, F1 0.857 |

Key design choice in both tasks: **all splits are done by patient**, so recordings of the same subject never appear in both train and test (no data leakage).

---

## Task 1: Feature-based pipeline

**Dataset:** [Gait in Parkinson's Disease (PhysioNet)](https://physionet.org/content/gaitpdb/1.0.0/), foot-pressure sensor recordings of PD patients and healthy controls.

**Pipeline**
1. **Preprocessing:** low-pass Butterworth filter (`scipy.signal.butter` + `filtfilt`, zero-phase).
2. **Gait cycle segmentation** and **feature extraction**:
   - heel and toe peak forces (mean, SD);
   - gait-cycle shape statistics (skewness, kurtosis);
   - spectral features (PSD peak power).
3. **Statistical analysis:** t-test per feature, PD vs. control.
4. **Models:** `StandardScaler` + SVM (RBF) and Random Forest, with `class_weight="balanced"` to handle class imbalance.
5. **Validation:** `GroupKFold` by subject (5 folds), hyperparameters tuned with nested `GridSearchCV`.
6. **Interpretation:** permutation importance and RF feature importances.

**Results (cross-validation, mean ± std)**

| Metric | SVM RBF | Random Forest |
|--------|---------|---------------|
| Accuracy | 0.884 ± 0.085 | 0.875 ± 0.061 |
| Balanced Accuracy | **0.860 ± 0.105** | 0.826 ± 0.085 |
| F1-score | **0.917 ± 0.061** | 0.913 ± 0.040 |
| ROC-AUC | **0.958 ± 0.044** | 0.956 ± 0.026 |

SVM is slightly better on balanced accuracy (+3.4 p.p.), which matters under class imbalance. Random Forest is more stable (lower std) and interpretable out of the box.

---

## Task 2: Neural network on accelerometry

**Dataset:** [PD-BioStampRC21 (IEEE DataPort)](https://ieee-dataport.org/open-access/pd-biostamprc21-parkinsons-disease-accelerometry-dataset-five-wearable-sensor-study-0), 3-axis accelerometers from 5 wearable sensors, 31.25 Hz.

**Pipeline**
1. Sensors are synchronized and stacked into a multichannel signal (5 × 3 axes).
2. **Sliding windows:** 10 s length, 5 s stride.
3. **Patient-level split:** train / val / test = 60 / 20 / 20 %, stratified by diagnosis.
4. **Augmentation (train only):** Gaussian noise + random amplitude scaling.
5. **Model: CNN-BiLSTM**
   - 3 × [Conv1d → BatchNorm → ReLU → MaxPool], 64 → 128 → 256 channels, for local motion patterns;
   - bidirectional LSTM (128 hidden units) for temporal dependencies;
   - mean pooling over time → MLP head with Dropout 0.5.
6. **Training:** Adam (lr 3e-4), class-weighted cross-entropy, 50 epochs, best checkpoint chosen on validation.
7. **Patient-level inference:** window probabilities are averaged per subject.

**Results (test set)**

| Level | F1 | ROC-AUC |
|-------|----|---------|
| Windows | 0.774 | 0.817 |
| Patients | **0.857** | **0.889** |

Averaging over windows smooths noise and noticeably improves subject-level separation.

---

## Limitations
- Class imbalance, handled with class weights and balanced metrics.
- The tasks use different datasets and sensors, so their metrics are not directly comparable. On small datasets, interpretable hand-crafted features remain a strong baseline compared to end-to-end networks.

## Repository structure
```
task1/
  svm.ipynb              # Task 1: features + SVM
  random_forest.ipynb    # Task 1: features + Random Forest
pd_project/              # Task 2
  run.py                 # CLI: train / eval
  visualize.py           # ROC, PR, confusion matrices, per-patient bars
  src/dataset.py         # loading, windowing, patient split, augmentation
  src/model.py           # CNN-BiLSTM
  src/train.py, eval.py, utils.py
  checkpoints/best_model.pt
  figures_val/, figures_test/
```

## How to run (Task 2)
```bash
cd pd_project
pip install -r requirements.txt
python run.py train --data_root /path/to/data --clinic_file Clinic_DataPDBioStampRCStudy.csv \
  --window_sec 10 --stride_sec 5 --batch_size 8 --lr 3e-4 --epochs 50 --val_size 0.2 --test_size 0.2
python run.py eval  --data_root /path/to/data --clinic_file Clinic_DataPDBioStampRCStudy.csv
python visualize.py --data_root /path/to/data --clinic_file Clinic_DataPDBioStampRCStudy.csv --split test --out_dir figures
```
Detailed instructions are in [pd_project/README.md](pd_project/README.md). The datasets are not included; download them from the links above.

## Stack
Python, NumPy, pandas, SciPy, scikit-learn, PyTorch, Matplotlib, Jupyter

## Context
Course project at Bauman Moscow State Technical University (Biomedical Engineering), 2026.
