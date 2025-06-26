# Comparative Analysis of Machine Learning Techniques for Credit Card Fraud Detection
=============================================================================================================================

## Abstract

This project explores and compares various machine learning strategies for handling highly imbalanced datasets, using credit card fraud detection as a case study. The analysis focuses on a well-known dataset where fraudulent transactions account for less than 0.2% of the data. We investigate several approaches, including statistical outlier detection (Mahalanobis Distance), unsupervised novelty detection (One-Class SVM), and a suite of supervised learning methods. Key supervised techniques include cost-sensitive learning with a class-weighted Support Vector Machine (SVM) and hybrid data sampling (SMOTEENN) to train multiple classifiers (SVM, Decision Tree, Naïve Bayes, KNN). Finally, these methods are benchmarked against an XGBoost ensemble model. The results demonstrate that while data sampling improves fraud recall, it significantly degrades precision. The XGBoost model, trained on the original imbalanced data, ultimately provides the best balance of precision and recall, proving to be the most robust solution.
=============================================================================================================================

#H1 Project Overview
This repository provides a complete, executable implementation of the comparative analysis described in the paper "Imbalance Dataset Approaches and Supervised Learning Comparative Analysis: Credit Card Fraud Detection." The primary goal is to determine the most effective machine learning strategy for identifying fraudulent credit card transactions from a dataset with a severe class imbalance.
============================================================================================================v================

#Key Questions Explored:
How do unsupervised, statistical, and supervised methods compare in fraud detection?
Is it more effective to use class-weighting to penalize errors on the minority class or to resample the data to be balanced?
Can a powerful ensemble method like XGBoost outperform other models on either the original or a balanced dataset?


#Data Source
The study uses the public "Credit Card Fraud Detection" dataset, originally from Kaggle. The dataset contains transactions made by European cardholders in September 2013.

It consists of 284,807 transactions, of which only 492 (0.172%) are fraudulent.
To protect user privacy, the original features have been transformed via Principal Component Analysis (PCA) into 28 numerical features (V1 to V28).

The only features that remain untransformed are Time (seconds elapsed since the first transaction) and Amount (transaction amount).
The target variable is Class, where 1 indicates fraud and 0 indicates a legitimate transaction.
A copy of the dataset (creditcard.csv) is included in this repository.

#Methodology & Findings
The project systematically evaluates different families of algorithms:

#Statistical & Unsupervised Methods:
##Mahalanobis Distance: A statistical method to detect multivariate outliers. It showed poor performance, struggling to distinguish the weak outliers present in this dataset.
One-Class SVM: An unsupervised algorithm for novelty detection. It also performed poorly, with an F1-score of only 0.02 for the fraud class.
##Supervised Methods: This was the main focus of the analysis.
##Cost-Sensitive Learning (Weighted SVM): By assigning a higher penalty to misclassifying fraud, the SVM's recall for fraud jumped to 85%. However, its precision was very low (8%), meaning it generated many false positives.
##Hybrid Data Sampling (SMOTEENN):
The training data was balanced using SMOTE (to create synthetic fraud examples) and Edited Nearest Neighbours (to remove ambiguous majority-class examples).
Models trained on this balanced data (SVM, Decision Tree, KNN, Naïve Bayes) generally showed high recall for fraud but suffered from low precision.
##Ensemble Learning (XGBoost):
When trained on the original imbalanced data, XGBoost was the clear winner. It achieved an excellent AUC of 0.985 and a high F1-score for fraud detection while maintaining high precision on normal transactions.
Training XGBoost on the sampled data slightly improved fraud recall but significantly worsened precision, confirming that sampling introduced a negative bias for this powerful model.

#Conclusion
The analysis concludes that for this dataset, XGBoost trained on the original, imbalanced data is the superior approach. It demonstrates that modern gradient boosting models are robust enough to handle severe class imbalance without requiring data resampling, which can introduce unwanted biases and increase false positives.

Corrections and Code Notes
The Python code has been consolidated from the paper's appendix and a supplementary notebook into a single, clean, and executable script (fraud_detection_analysis.py).

Outdated libraries like unicodecsv have been replaced with standard pandas.
Redundant code for plotting and evaluation has been refactored into reusable functions for clarity.

The script focuses on the most critical comparisons discussed in the paper: the class-weighted SVM, the hybrid-sampled SVM, and the XGBoost models.

How to Run the Analysis
Clone the Repository:

```git
git clone <repository-url>
cd <repository-directory>
```

#Install Dependencies: Make sure you have the required Python packages installed.
#pip install pandas scikit-learn matplotlib seaborn imbalanced-learn xgboost scikit-optimize
#Place Data File: Ensure creditcard.csv is in the same directory as the Python script.
#Execute the Script: Run the analysis from your terminal.

#python fraud_detection_analysis.py

The script will perform the data analysis, train the models, print the evaluation metrics to the console, and display the resulting confusion matrices and ROC curves.
