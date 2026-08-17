# MNIST and Letter Recognition with K-Nearest Neighbors

This consolidated repository combines the relevant MNIST and Letter Recognition KNN work from `MNIST-DATASET`, `Letter-recognitiion`, and `MNIST-and-Letter-Recognition-Dataset-Classification`.

## Project objective

The project studies how K-Nearest Neighbors behaves across handwritten-digit and alphabet-character classification tasks. It compares distance metrics, evaluates different K values, and preserves the result files and report generated during the experiments.

## Contents

```text
notebooks/
  mnist_knn_comparison.ipynb        MNIST KNN experiment and metric comparison
  letter_recognition_knn.ipynb      Letter Recognition KNN experiment
results/
  knn_chebyshev_results.xlsx
  knn_euclidean_results.csv
  knn_manhattan_results.csv
  knnmnist_comparison_results.csv
  knnletterrecognition_results.xlsx
  mnist_knn_report.docx
```

## Topics covered

- KNN classification on MNIST and Letter Recognition data.
- Euclidean, Manhattan, and Chebyshev distance comparisons.
- Evaluation across multiple K values.
- Accuracy and confusion-matrix analysis.
- Exported result tables for comparing experiments.

## Run the notebooks

```bash
pip install jupyter notebook numpy pandas scikit-learn matplotlib seaborn openpyxl
jupyter notebook
```

Open the notebooks under `notebooks/` and run the cells in order. Review each notebook's dataset-loading section because the original experiments may use downloaded or notebook-specific dataset paths.

## Consolidation note

The original repositories contained overlapping MNIST and Letter Recognition notebooks. The canonical version keeps one MNIST experiment, one Letter Recognition experiment, the unique result exports, and the original report. Repeated repository-level READMEs and a duplicate Letter Recognition experiment were excluded from the consolidated staging copy.
