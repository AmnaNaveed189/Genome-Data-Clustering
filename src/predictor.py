# src/predictor.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_subtype_predictor(X_pca, labels, config):
    """
    Train a classifier that can predict the subtype
    of a NEW patient not seen during clustering.

    This converts unsupervised → supervised for deployment.

    Args:
        X_pca: PCA-reduced features
        labels: cluster labels from K-Means
        config: project config
    Returns:
        clf: trained classifier
        test metrics
    """
    print("Training subtype predictor...")

    n_clusters = len(np.unique(labels))
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[f'Subtype {i}' for i in range(n_clusters)]
    ))

    # Cross-validation
    cv_scores = cross_val_score(clf, X_pca, labels, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[f'Subtype {i}' for i in range(n_clusters)],
        yticklabels=[f'Subtype {i}' for i in range(n_clusters)]
    )
    plt.title("Confusion Matrix — Subtype Predictor")
    plt.ylabel("True Subtype")
    plt.xlabel("Predicted Subtype")
    plt.tight_layout()
    plt.savefig('results/11_confusion_matrix.png', dpi=150)
    plt.show()

    return clf

def save_all_models(scaler, pca, kmeans, clf, save_dir='models/'):
    """Save all pipeline models for deployment"""
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(scaler, f'{save_dir}/scaler.pkl')
    joblib.dump(pca,    f'{save_dir}/pca.pkl')
    joblib.dump(kmeans, f'{save_dir}/kmeans.pkl')
    joblib.dump(clf,    f'{save_dir}/classifier.pkl')

    print(f"All models saved to {save_dir}/")
    print("  scaler.pkl     — preprocessing scaler")
    print("  pca.pkl        — PCA reducer")
    print("  kmeans.pkl     — clustering model")
    print("  classifier.pkl — subtype predictor")

def predict_new_patient(gene_expression_values, models_dir='models/'):
    """
    Predict the molecular subtype for a new patient.

    Args:
        gene_expression_values: list/array of gene expression values
    Returns:
        subtype: predicted cluster number
        confidence: prediction probability
    """
    scaler = joblib.load(f'{models_dir}/scaler.pkl')
    pca    = joblib.load(f'{models_dir}/pca.pkl')
    clf    = joblib.load(f'{models_dir}/classifier.pkl')

    x = np.log1p(np.array(gene_expression_values))
    x = scaler.transform([x])
    x = pca.transform(x)

    subtype = clf.predict(x)[0]
    confidence = clf.predict_proba(x)[0].max()

    return int(subtype), float(confidence)