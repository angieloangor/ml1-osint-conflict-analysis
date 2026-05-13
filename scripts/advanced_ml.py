#!/usr/bin/env python3
"""
Advanced ML Module
==================

Entrena modelos supervisados avanzados: Random Forest, XGBoost.
Incluye class_weight, stratified split, y comparación comprehensiva.

Outputs:
- outputs/model_comparison_advanced.csv: Comparación de todos los modelos
- outputs/advanced_figures/model_comparison_advanced.png
- outputs/models/confusion_matrix_*.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings('ignore')

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load labeled dataset and TF-IDF features."""
    df = pd.read_csv('data/dataset_nlp_labeled.csv', parse_dates=['timestamp'])
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(df['processed_text'].fillna(''))
    
    return df, X

def stratified_split(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Perform stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Train model and compute metrics."""
    print(f'Training {model_name}...')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    print(f'  Accuracy: {metrics["accuracy"]:.4f}')
    print(f'  F1 Macro: {metrics["f1_macro"]:.4f}')
    
    return metrics, y_pred, model

def get_class_weights(y: pd.Series) -> dict:
    """Compute class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {cls: w for cls, w in zip(classes, weights)}

def plot_confusion_matrices(
    y_test: pd.Series,
    predictions: dict,
    output_dir: Path | str = 'outputs/advanced_figures'
) -> None:
    """Plot confusion matrices for all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classes = sorted(y_test.unique())
    n_classes = len(classes)
    
    for model_name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved confusion matrix: {output_dir / f"confusion_matrix_{model_name.lower()}.png"}')

def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path | str = 'outputs/advanced_figures/model_comparison_advanced.png') -> None:
    """Plot model comparison."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax = axes[0]
    comparison_df.sort_values('accuracy', ascending=False).plot(
        x='model', y='accuracy', kind='barh', ax=ax, color='steelblue', legend=False
    )
    ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # F1 Macro
    ax = axes[1]
    comparison_df.sort_values('f1_macro', ascending=False).plot(
        x='model', y='f1_macro', kind='barh', ax=ax, color='coral', legend=False
    )
    ax.set_xlabel('F1 Macro', fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title('Model F1 Macro Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved model comparison: {output_path}')
    plt.close()

def run_advanced_ml_pipeline() -> dict:
    """Run advanced ML pipeline."""
    print('=' * 70)
    print('ADVANCED ML PIPELINE')
    print('=' * 70)
    
    df, X = load_data()
    y = df['weak_label']
    
    print(f'\nDataset: {len(df)} samples, {X.shape[1]} features')
    print(f'Classes: {y.value_counts().to_dict()}')
    
    # Stratified split
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f'\nTrain: {len(y_train)}, Test: {len(y_test)}')
    
    # Get class weights
    class_weights = get_class_weights(y_train)
    print(f'Class weights: {class_weights}')
    
    # Train models
    results = []
    predictions = {}
    
    models = [
        (LogisticRegression(max_iter=1000, class_weight='balanced'), 'LogisticRegression'),
        (MultinomialNB(), 'MultinomialNB'),
        (KNeighborsClassifier(n_neighbors=5), 'KNeighborsClassifier'),
        (RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), 'RandomForest'),
        (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42), 'GradientBoosting'),
    ]
    
    for model, model_name in models:
        metrics, y_pred, trained_model = train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name)
        results.append(metrics)
        predictions[model_name] = y_pred
    
    # Compile results
    comparison_df = pd.DataFrame(results)
    comparison_path = Path('outputs/model_comparison_advanced.csv')
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    print(f'\nSaved model comparison: {comparison_path}')
    print(comparison_df.to_string(index=False))
    
    # Visualizations
    plot_confusion_matrices(y_test, predictions)
    plot_model_comparison(comparison_df)
    
    summary = {
        'n_samples': len(df),
        'n_features': X.shape[1],
        'n_classes': len(y.unique()),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'n_models': len(models),
        'best_model': comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model'],
        'best_accuracy': comparison_df['accuracy'].max(),
        'comparison_path': str(comparison_path)
    }
    
    print('\nAdvanced ML pipeline completed')
    for key, value in summary.items():
        print(f'  {key}: {value}')
    
    return summary

if __name__ == '__main__':
    run_advanced_ml_pipeline()
