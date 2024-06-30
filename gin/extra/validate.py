import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tonic.validate import plot_threshold_results  # Assuming this is where the function is located
from torch_geometric.data import DataLoader
import pandas as pd

def evaluate_thresholds_gnn(model, test_data, thresholds)->pd.DataFrame:
    model.eval()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(batch.y.numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = []
    for threshold in thresholds:
        y_pred = (all_preds >= threshold).astype(int)
        
        precision, recall, _ = precision_recall_curve(all_labels, y_pred)
        fpr, tpr, _ = roc_curve(all_labels, y_pred)
        
        pr_auc = auc(recall, precision)
        roc_auc = auc(fpr, tpr)
        
        accuracy = np.mean(y_pred == all_labels)
        precision = np.sum((y_pred == 1) & (all_labels == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == 1) & (all_labels == 1)) / np.sum(all_labels == 1)
        prec0 = np.sum((y_pred == 0) & (all_labels == 0)) / np.sum(y_pred == 0)
        rec0 = np.sum((y_pred == 0) & (all_labels == 0)) / np.sum(all_labels == 0)
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score0 = 2 * (prec0 * rec0) / (prec0 + rec0)

        
        results.append({
            'Threshold': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1_score,
            'F1_0': f1_score0,
            'Pr_auc': pr_auc,
            'Roc_auc': roc_auc
        })

    return pd.DataFrame(results)

