# Comparative Analysis of Machine Learning Techniques for Credit Card Fraud Detection

  

## Abstract

This project explores and compares various machine learning strategies for handling highly imbalanced datasets, using credit card fraud detection as a case study. The analysis focuses on a well-known dataset where fraudulent transactions account for less than 0.2% of the data. We investigate several approaches, including statistical outlier detection (Mahalanobis Distance), unsupervised novelty detection (One-Class SVM), and a suite of supervised learning methods. 

Key supervised techniques include cost-sensitive learning with a class-weighted Support Vector Machine (SVM) and hybrid data sampling (SMOTEENN) to train multiple classifiers (SVM, Decision Tree, Naïve Bayes, KNN). Finally, these methods are benchmarked against an XGBoost ensemble model. The results demonstrate that while data sampling improves fraud recall, it significantly degrades precision. The XGBoost model, trained on the original imbalanced data, ultimately provides the best balance of precision and recall, proving to be the most robust solution.



## Project Overview:

This repository provides a complete, executable implementation of the comparative analysis described in the paper "Imbalance Dataset Approaches and Supervised Learning Comparative Analysis: Credit Card Fraud Detection." The primary goal is to determine the most effective machine learning strategy for identifying fraudulent credit card transactions from a dataset with a severe class imbalance.



## Key Questions Explored:

 * How do unsupervised, statistical, and supervised methods compare in fraud detection?
 * Is it more effective to use class-weighting to penalize errors on the minority class or to resample the data to be balanced?
 * Can a powerful ensemble method like XGBoost outperform other models on either the original or a balanced dataset?



## Data Source

 * The study uses the public "Credit Card Fraud Detection" dataset, originally from Kaggle. The dataset contains transactions made by European cardholders in September 2013.
 * It consists of 284,807 transactions, of which only 492 (0.172%) are fraudulent.
 * To protect user privacy, the original features have been transformed via Principal Component Analysis (PCA) into 28 numerical features (V1 to V28).
 * The only features that remain untransformed are Time (seconds elapsed since the first transaction) and Amount (transaction amount).
 * The target variable is Class, where 1 indicates fraud and 0 indicates a legitimate transaction.
 * The dataset file (`creditcard.csv`, ~150 MB) is **not** included in this repository. Download it from Kaggle ("Credit Card Fraud Detection", a free Kaggle login is required) and place it next to the script.



## Methodology & Findings

### The project systematically evaluates different families of algorithms:

#### Statistical & Unsupervised Methods:

  * Mahalanobis Distance: A statistical method to detect multivariate outliers. It showed poor performance, struggling to distinguish the weak outliers present in this dataset.
  * One-Class SVM: An unsupervised algorithm for novelty detection. It also performed poorly, with an F1-score of only 0.02 for the fraud class.

#### Supervised Methods: This was the main focus of the analysis.

  * Cost-Sensitive Learning (Weighted SVM): By assigning a higher penalty to misclassifying fraud, the SVM's recall for fraud jumped to 85%. However, its precision was very low (8%), meaning it generated many false positives.

#### Hybrid Data Sampling (SMOTEENN):

  *  The training data was balanced using SMOTE (to create synthetic fraud examples) and Edited Nearest Neighbours (to remove ambiguous majority-class examples).
  *  Models trained on this balanced data (SVM, Decision Tree, KNN, Naïve Bayes) generally showed high recall for fraud but suffered from low precision.

#### Ensemble Learning (XGBoost):

  *  When trained on the original imbalanced data, XGBoost was the clear winner. It achieved an excellent AUC of 0.985 and a high F1-score for fraud detection while maintaining high precision on normal transactions.
  * Training XGBoost on the sampled data slightly improved fraud recall but significantly worsened precision, confirming that sampling introduced a negative bias for this powerful model.

## Conclusion

The analysis concludes that for this dataset, XGBoost trained on the original, imbalanced data is the superior approach. It demonstrates that modern gradient boosting models are robust enough to handle severe class imbalance without requiring data resampling, which can introduce unwanted biases and increase false positives.

Corrections and Code Notes
The Python code has been consolidated from the paper's appendix and a supplementary notebook into a single, clean, and executable script (fraud_detection_analysis.py).

Outdated libraries like unicodecsv have been replaced with standard pandas.
Redundant code for plotting and evaluation has been refactored into reusable functions for clarity.

The script focuses on the most critical comparisons discussed in the paper: the class-weighted SVM, the hybrid-sampled SVM, and the XGBoost models.

## How to run

Tested with Python 3.11 (see `requirements.txt` for the pinned versions).

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put creditcard.csv (from Kaggle) in this folder, then:
python fraud_detection_analysis.py                      # prints metrics, shows confusion matrices and ROC curves
jupyter nbconvert --to notebook --execute notebook_alternative.ipynb --output out.ipynb   # the exploratory notebook
```

* Without a display, run the script with `MPLBACKEND=Agg` so the plots do not block.
* The script runs three experiments (class-weighted SVM, SMOTEENN + SVM, XGBoost). The One-Class SVM and Mahalanobis experiments from the paper are not in the script.
* The notebook is the original exploratory work and is slow on the full dataset (the saved outputs show runs of up to ~1 hour per cell). Its saved outputs come from the original run.

## About the numbers in this README

All figures above (AUC 0.985, SVM recall 85% / precision 8%, One-Class SVM F1 0.02, ...) are taken from the tables of the coursework report in this repository (`9870776MLCoursework.pdf`). They were **not** re-produced when the code was last fixed, because `creditcard.csv` is not in the repository. After the fix the script and notebook were only smoke-tested on a small randomly generated table with the same columns (no meaningful metrics). The script uses its own 70/30 stratified split, so its output may differ from the report's tables.

## Corrections and Code Notes

The Python code has been consolidated from the paper's appendix and a supplementary notebook into a single executable script (`fraud_detection_analysis.py`).

* `unicodecsv` was replaced with pandas, and redundant plotting/evaluation code was refactored into reusable functions.
* The script focuses on the most critical comparisons discussed in the paper: the class-weighted SVM, the hybrid-sampled SVM, and the XGBoost model.
* The paper has the original code in its appendix.
