import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc

from typing import Tuple

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray|None = None):
    """Evaluate the model performance and print metrics."""
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
    print("Precision:", metrics.precision_score(y_true, y_pred))
    print("Recall:", metrics.recall_score(y_true, y_pred))
    print("F1 Score:", metrics.f1_score(y_true, y_pred))
    print("F1 Score0:", metrics.f1_score(y_true, y_pred, pos_label=0))
    print("AUC-ROC Score:", metrics.roc_auc_score(y_true, y_prob) if y_prob is not None else "N/A")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list = ['Non-floral', 'Floral'], suptitle=''):
    """Plot the confusion matrix."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # let's add F1-score and F1-score0 to the title
    f1 = metrics.f1_score(y_true, y_pred)
    f0 = metrics.f1_score(y_true, y_pred, pos_label=0)
    plt.title('Confusion Matrix' + f'\nF1-score: {f1:.2f}, F1-score0: {f0:.2f}')
    if suptitle:
        plt.suptitle(suptitle)
    plt.show()

def evaluate_thresholds(model, X_test, y_test, thresholds,
                        y_proba: np.ndarray|None = None)->pd.DataFrame:
    """
    Evaluate different thresholds for a given model.
    
    Args:
    - model: Trained model with predict_proba method.
    - X_test: Test features.
    - y_test: True labels for test set.
    - thresholds: List or array of thresholds to evaluate.
    
    Returns:
    - DataFrame with threshold, accuracy, precision, recall, and F1 score for each threshold.
    """
    results = []
    if y_proba is None:
        predicted_proba = model.predict_proba(X_test)[:, 1]
    else:
        predicted_proba = y_proba
    
    for threshold in thresholds:
        predicted = (predicted_proba >= threshold).astype('int')
        accuracy = accuracy_score(y_test, predicted)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average='binary')
        f1_0 = metrics.f1_score(y_test, predicted, pos_label=0)
        results.append((threshold, accuracy, precision, recall, f1, f1_0))
    
    results_df = pd.DataFrame(results, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'F1_0'])
    return results_df

def plot_threshold_results(results_df, model_name, suptitle=''):
    """
    Plot the results of threshold evaluation.
    
    Args:
    - results_df: DataFrame with threshold, accuracy, precision, recall, and F1 score.
    - model_name: Name of the model (for plot title).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['Accuracy'], label='Accuracy', linestyle=':')
    plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision', linestyle=':')
    plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall', linestyle=':')
    plt.plot(results_df['Threshold'], results_df['F1'], label='F1', linewidth=2)
    plt.plot(results_df['Threshold'], results_df['F1_0'], label='F1_0', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold vs Performance for {model_name}')
    # verticle line at best threshold
    best_threshold = results_df.loc[results_df['F1'].idxmax(), 'Threshold']
    best_threshold0 = results_df.loc[results_df['F1_0'].idxmax(), 'Threshold']
    plt.axvline(x=best_threshold, color='black', linestyle='--', label='Best Threshold-1')
    plt.axvline(x=best_threshold0, color='black', linestyle='--', label='Best Threshold-0')
    plt.axvspan(best_threshold0, best_threshold, color='gray', alpha=0.2)
    plt.legend()
    if suptitle:
        plt.suptitle(suptitle)
    plt.show()

def plot_roc_curve(y_true, y_proba, suptitle=None,
                   annotate_thresholds: bool = False)->pd.DataFrame:
    """
    Plot the ROC curve.
    
    Args:
    - y_true: True labels.
    - y_proba: Predicted probabilities.
    - suptitle: Title for the plot (optional).
    
    Returns:
    - fpr: False positive rate.
    - tpr: True positive rate.
    - threshold: Thresholds used to compute fpr and tpr.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, threshold = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.scatter(fpr, tpr, c='darkorange', label='Thresholds', s=50)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # text of thresholds on the plot
    if annotate_thresholds:
        for i, txt in enumerate(threshold):
            plt.annotate(f'{txt:.2f}', (fpr[i], tpr[i]), fontsize=8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    if suptitle:
        plt.suptitle(suptitle)

    # Optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    plt.axvline(x=fpr[optimal_idx], color='r', linestyle='--')
    plt.axhline(y=tpr[optimal_idx], color='r', linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='r', s=200, label='Optimal Threshold')
    plt.legend(loc='lower right')
    plt.show()
    
    return pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold, 'optimal_idx': optimal_idx == np.arange(len(threshold))})

# Example usage
if __name__ == "__main__":
    # Simulated test data
    y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.2, 0.85, 0.75, 0.6])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.2, 0.85, 0.75, 0.6])
    
    # Evaluate model
    evaluate_model(y_test, y_pred, y_prob)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_proba, annotate_thresholds=True)
