# A Reproducible Machine-Learning Benchmark for Cancer Classification

## Overview

This project benchmarks three machine-learning classifiers for breast cancer classification using the Wisconsin Diagnostic Breast Cancer dataset available through scikit-learn. It focuses on reproducible evaluation, cross-validation, and whether tuning a model to prioritize malignant-case recall improves performance on a held-out test set.

## Research questions

- How do Logistic Regression, Support Vector Machine (SVM), and Random Forest compare on this dataset?
- How consistent are their results under stratified cross-validation?
- Does tuning Logistic Regression to prioritize malignant-case recall improve held-out test performance?

## Dataset

The project uses `sklearn.datasets.load_breast_cancer`, which contains:

- 569 samples
- 30 numerical features
- Two classes: malignant and benign
- No missing values identified during the initial data inspection

In the dataset's original target coding, `0` represents malignant and `1` represents benign. The analysis accounts for this when calculating malignant-class metrics.

## Methods

1. **Exploratory data analysis:** inspect class balance, feature summaries, and feature relationships.
2. **Train/test split:** use a stratified 80:20 split with `random_state=42` (455 training samples and 114 test samples).
3. **Baseline models:** evaluate Logistic Regression, SVM, and Random Forest. Standardize features for Logistic Regression and SVM; use 300 trees and balanced class weights for Random Forest.
4. **Cross-validation:** compare models using 5-fold stratified cross-validation, including accuracy, malignant recall, and ROC-AUC.
5. **Hyperparameter tuning:** use `GridSearchCV` to tune Logistic Regression's regularization strength and class weighting, with malignant recall as the optimization metric.
6. **Nested cross-validation:** compare tuned and baseline approaches using 5 outer folds and 5 inner folds.
7. **Error analysis and final evaluation:** examine errors and compare models on the held-out test set.

## Held-out test-set results

| Model | Accuracy | Malignant recall | Malignant precision | Malignant F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9954 |
| SVM (baseline) | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9950 |
| Random Forest (baseline) | 0.9386 | 0.9286 | 0.9070 | 0.9176 | 0.9940 |
| Logistic Regression (tuned) | 0.9561 | 0.9762 | 0.9111 | 0.9425 | 0.9954 |

Metrics are rounded to four decimal places. Values are calculated from the saved final test-set comparison: [`final_test_set_comparison.csv`](Results/final_test_set_comparison.csv).

## Main findings

- Baseline Logistic Regression and SVM achieved the highest held-out test accuracy, at 98.25%.
- Baseline and tuned Logistic Regression both achieved 97.62% malignant recall on the held-out test set.
- Tuning Logistic Regression to prioritize malignant recall did not improve held-out malignant recall.
- The tuned Logistic Regression model had lower held-out accuracy and malignant precision than the baseline.
- Nested cross-validation did not show an improvement over baseline Logistic Regression in this benchmark.

These results describe performance on this dataset and should not be interpreted as evidence of clinical readiness.

## Confusion matrix

The following figure is the saved final Logistic Regression confusion matrix:

![Final Logistic Regression confusion matrix](Results/final_logistic_regression_confusion_matrix.png)

The additional [Logistic Regression confusion matrix](Results/logistic_regression_confusion_matrix.png) is also available in the results folder.

## Repository structure

```text
Breast-Cancer-ML-Benchmark/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── README.md
│   └── 01_cancer_classification_benchmark.ipynb
└── Results/
    ├── README.md
    ├── cross_validation_results.csv
    ├── logistic_regression_tuning_comparison.csv
    ├── nested_cv_model_comparison.csv
    ├── final_test_set_comparison.csv
    ├── logistic_regression_confusion_matrix.png
    └── final_logistic_regression_confusion_matrix.png
```

## Reproducibility

The notebook was tested with Python 3.13. It uses NumPy, pandas, Matplotlib, Seaborn, scikit-learn, and Jupyter Notebook. Random seeds are set for reproducibility.

To reproduce the analysis:

1. Install the dependencies listed in `requirements.txt`.
2. Open `notebooks/01_cancer_classification_benchmark.ipynb` in Jupyter Notebook.
3. Restart the kernel and run all cells from top to bottom.

## Limitations

- The analysis uses one established dataset with 569 samples.
- The held-out test set contains 114 samples, so individual errors can affect the reported metrics.
- Cross-validation and test-set performance do not guarantee generalization to new populations or clinical settings.
- This is a machine-learning benchmark, not a clinical decision-support tool.

## Conclusion

This project demonstrates a reproducible classification workflow combining exploratory analysis, baseline model comparison, stratified cross-validation, hyperparameter tuning, nested cross-validation, and held-out test evaluation. In this experiment, tuning Logistic Regression to prioritize malignant recall did not improve held-out malignant recall compared with baseline Logistic Regression.
