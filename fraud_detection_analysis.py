# -----------------------------------------------------------------------------
# COMPARATIVE ANALYSIS OF MACHINE LEARNING FOR CREDIT CARD FRAUD DETECTION
# -----------------------------------------------------------------------------
# This script reproduces the core experiments from the paper "Imbalance Dataset
# Approaches and Supervised Learning Comparative Analysis: Credit Card Fraud
# Detection".
#
# It compares three main strategies:
# 1. Cost-Sensitive Learning: A class-weighted SVM.
# 2. Hybrid Sampling: An SVM trained on data balanced with SMOTEENN.
# 3. Ensemble Learning: An XGBoost classifier.
#
# HOW TO USE:
# 1. Ensure all required packages are installed (pandas, scikit-learn,
#    imbalanced-learn, xgboost, matplotlib, seaborn, scikit-optimize).
# 2. Place 'creditcard.csv' in the same directory as this script.
# 3. Run the script from the command line: python fraud_detection_analysis.py
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and Sampling
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

# Models
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    log_loss,
    accuracy_score
)

# Hyperparameter Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

# --- Configuration ---
# Set random state for reproducibility
RANDOM_STATE = 42

# --- Utility Functions for Evaluation and Plotting ---

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plots a confusion matrix using seaborn's heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral_r',
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

def plot_roc_curve(y_true, y_probs, model_name):
    """Plots the ROC curve for a given model."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a trained model and prints a comprehensive report."""
    print(f"\n--- Evaluating {model_name} ---")
    start_time = time.time()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    end_time = time.time()
    
    # Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    
    print(f"Prediction Time: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log-Loss: {loss:.4f}")
    print(f"AUC Score: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))
    
    # Plotting
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_probs, model_name)

# --- Main Analysis Script ---

def main():
    """Main function to run the credit card fraud detection analysis."""
    # 1. Load and Prepare Data
    print("1. Loading and preparing data...")
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: 'creditcard.csv' not found. Please place the dataset in the correct directory.")
        return

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Scale 'Amount' and 'Time' features
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Define features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data into training and testing sets (70/30 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # 2. Experiment 1: Cost-Sensitive Learning (Class-Weighted SVM)
    print("\n2. Training Class-Weighted SVM...")
    start_time = time.time()
    
    # Define hyperparameter search space for Bayesian Optimization
    svm_params = {
        'C': Real(1e-3, 1e+3, prior='log-uniform'),
        'gamma': Real(1e-4, 1e+2, prior='log-uniform'),
        'kernel': Categorical(['rbf']),
    }
    
    weighted_svm = SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE)
    
    # Using Bayesian Search for more efficient hyperparameter tuning than Grid Search
    bayes_search_svm = BayesSearchCV(
        estimator=weighted_svm,
        search_spaces=svm_params,
        n_iter=10,  # Reduced iterations for speed; increase for better tuning
        cv=3,
        n_jobs=-1,
        scoring='roc_auc',
        random_state=RANDOM_STATE
    )
    bayes_search_svm.fit(X_train, y_train)
    best_weighted_svm = bayes_search_svm.best_estimator_
    
    end_time = time.time()
    print(f"Weighted SVM training and tuning time: {end_time - start_time:.2f} seconds")
    print(f"Best parameters found: {bayes_search_svm.best_params_}")

    evaluate_model(best_weighted_svm, X_test, y_test, "Class-Weighted SVM")

    # 3. Experiment 2: Hybrid Sampling (SMOTEENN) + SVM
    print("\n3. Performing Hybrid Sampling (SMOTEENN) and training SVM...")
    start_time_sampling = time.time()
    
    smote_enn = SMOTEENN(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    
    end_time_sampling = time.time()
    print(f"SMOTEENN resampling time: {end_time_sampling - start_time_sampling:.2f} seconds")
    print(f"Original training shape: {Counter(y_train)}")
    print(f"Resampled training shape: {Counter(y_train_resampled)}")

    # Train a standard SVM on the resampled data
    start_time_svm_sampled = time.time()
    
    # We can reuse the same search space
    standard_svm = SVC(probability=True, random_state=RANDOM_STATE)
    bayes_search_svm_sampled = BayesSearchCV(
        estimator=standard_svm,
        search_spaces=svm_params,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        scoring='roc_auc',
        random_state=RANDOM_STATE
    )
    bayes_search_svm_sampled.fit(X_train_resampled, y_train_resampled)
    best_sampled_svm = bayes_search_svm_sampled.best_estimator_
    
    end_time_svm_sampled = time.time()
    print(f"SVM on sampled data training and tuning time: {end_time_svm_sampled - start_time_svm_sampled:.2f} seconds")
    
    evaluate_model(best_sampled_svm, X_test, y_test, "SVM with SMOTEENN Sampling")

    # 4. Experiment 3: Ensemble Learning (XGBoost)
    print("\n4. Training XGBoost Classifier...")
    start_time_xgb = time.time()
    
    # Calculate scale_pos_weight for XGBoost's built-in class weighting
    # scale_pos_weight = count(negative examples) / count(positive examples)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )
    
    # No extensive tuning needed as defaults are strong, but this shows how it could be done
    # xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
    # For speed, we will use the well-performing defaults
    
    xgb.fit(X_train, y_train)
    
    end_time_xgb = time.time()
    print(f"XGBoost training time: {end_time_xgb - start_time_xgb:.2f} seconds")
    
    evaluate_model(xgb, X_test, y_test, "XGBoost Classifier")
    
    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    from collections import Counter
    main()

