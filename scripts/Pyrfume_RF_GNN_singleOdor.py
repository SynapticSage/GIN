# devintegration
#
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: GIN
#     language: python
#     name: in
# ---

# + [markdown] id="LGz0DB7-1f0c"
# Name: Ryan Young
#
# Date: 2024-05-15

# + [markdown] id="yejrQ-g50vYF"
# # Molecular Scent Analysis
# Hi!
#
# Welcome to this exploratory analysis notebook where we aim to predict whether a molecule might smell like a flower.
#
# This notebook's goal is to demonstrate an approach to solving a problem related to molecular scent prediction, with a focus on exploratory data analysis, model evaluation, and visualization of results.
#
# Towards the end, we will also explore some problems related to message passing and graph neural networks.

# + [markdown] id="dtaFLFKizYVN"
# # Dependencies
# The following cell imports necessary libraries and modules, including a private GitHub repo called `gin` where I developed the code for this analysis.
#
# Even with notebooks, I tend to modularize pieces into repos for:
# - Reproducibility
# - CI/C
# - Testing

# + id="030e98c5-f7f4-4811-98a0-fc97b9cc0ce3"
# Imports and Argparse
import importlib
import os
import shutil
import argparse
import datetime
import matplotlib.pyplot as plt
plt.ion()

import gin

import numpy as np
import torch
import pandas as pd  # Added pandas import
from sklearn import ensemble as sklearn_ensemble
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# New MLflow imports and initialization
import mlflow
from mlflow import log_params, log_metrics, log_artifacts, start_run, end_run, set_experiment
from gin.log.mlflow import describe_run

parser = argparse.ArgumentParser(
    description='Predict the presence of a specific odor descriptor.')
parser.add_argument('--archive', 
                    type=str, 
                    default='leffingwell',
                    help='Name of the Pyrfume data archive to use.')
parser.add_argument('--descriptor', 
                    type=str, 
                    default='floral',
                    help='The odor descriptor to predict.')
args = parser.parse_args(args=[])  # Added args=[] for notebook execution
desc = args.descriptor
args.script = "Pyrfume_RF_GNN_singleOdor.py"

print(" ------  ARGS -------- ")
print(args)
print(" --------------------- ")

# import seaborn as sns
plt.rcParams['figure.dpi'] = 150

# Save the figure
figure_dir = os.path.join(os.path.dirname(gin.__file__), '..', 'figures', args.descriptor)
print("Figure directory:", figure_dir)
os.makedirs(figure_dir, exist_ok=True)  # Create the directory if it doesn't exist
figure_path = (lambda x="": 
                os.path.join(figure_dir, f'{plt.gcf().get_suptitle() if not x else x}.png'))
dict_path = (lambda x="": os.path.abspath(os.path.join(figure_dir, "..",
                                                       f"{x}.json")))
df_path = os.path.join(figure_dir, "..", "df.csv")  # WARNING: in the face of more analyses, may have to split this dataframe
def save_fig(name="", fig=None):
    fig = fig or plt.gcf()
    mlflow.log_figure(figure=fig, 
                      artifact_file=name+".png")

# + [markdown] id="setup_mlflow"
# ## MLFLOW Setup
# This section sets up experiment tracking with MLflow.

# + id="setup_mlflow_code"
# Extract experiment name and run name using describe_run
exp_name, run_name = describe_run(args.__dict__,
                                  exp_name="Pyrfume_RF_GNN_singleOdor")
assert(not mlflow.active_run())
print("-"*50)
print("Experiment name:", )
print("Run name:", run_name)
print("-"*50)


def mlflow_annotate(kws:dict={}):
    kws.update(args.__dict__)
    mlflow.log_params(kws)
    mlflow.set_tag("stimuli_multi_or_single", "single")
    mlflow.set_tag("odorant_type", "pure_odorant")

# Set the MLflow experiment
mlflow.set_experiment(exp_name)
assert(not mlflow.active_run())

# + [markdown] id="d538e6fc"
# # Dataset
#
# Here we will use data managed by [the Pyrfume project](https://pyrfume.org/).
#
# The [SMILES strings](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) representing the molecular structures and their corresponding binary labels are provided.

# + id="load_data"
# Load the data
mlflow.start_run(run_name="exploration")
data_df = gin.data.pyrfume.get_join(args.archive, 
                                    types=["behavior", "molecules", "stimuli"])
data_df = pd.DataFrame(data_df.set_index('SMILES')[desc])

# + id="data_head"
data_df.head()

# + [markdown] id="af266b7c"
# Now that we have the data loaded, what should we learn about this dataset?

# + id="check_nulls"
data_df[desc].isnull().sum()

# + [markdown] id="2fd33ec1"
# No missing values - reassuring!
#
# Let's see the distribution of the labels in the dataset.

# + id="plot_distribution"
gin.explore.pyrfume.plot_desc_distribution(data_df, kind='pie', descriptor=desc)
save_fig(f'{desc}_distribution')

# Log metrics for class distribution
mlflow.log_dict({"class_distribution": data_df[desc].value_counts().to_dict()},
                artifact_file="class_distribution.json")

# + [markdown] id="d9d9222f"
# 👆 The large majority of the dataset is non-floral ❌💐. We should consider **class imbalance** downstream.

# + id="visualize_molecules"
# Let's visualize some of the molecular structures in the dataset and see if we can spot any patterns.
gin.explore.pyrfume.plot_molecular_structures_w_label(data_df, num_samples=20, descriptor=desc)
save_fig(f'{desc}_molecular_structures_1')

# + [markdown] id="2059e7bf"
# And let's examine a few more samples.

# + id="visualize_molecules_2"
gin.explore.pyrfume.plot_molecular_structures_w_label(data_df, num_samples=20, descriptor=desc)
save_fig(f'{desc}_molecular_structures_2')

# + [markdown] id="3b8a8363"
# ## Hypotheses
#
# Some things of note just from the visualization (hypotheses / possibilities / wild guesses):
#
# - The 🪻 floral molecules nearly all have oxygen with free electron pairs. Doubled-bond oxygen alone seems less often associated with floral molecules.
# - Nitrogen-containing molecules are rarely floral - though, devil's advocate, I also see fewer nitrogen-containing molecules to form an opinion.
#
# # Molecule Featurization
#
# In the next step, we will try to "digitize" each molecule by creating a 1D numpy array based on its molecular structure.

# + id="featurize_smiles"
def featurize_smiles(smiles_str: str,
                     method: str = 'combined') -> np.ndarray:
    """Convert a molecule SMILES into a 1D feature vector."""
    if method == 'morgan':
        fingerprint = gin.features.get_morgan_fingerprint(smiles_str)
    elif method == 'maccs':
        fingerprint = gin.features.get_maccs_keys_fingerprint(smiles_str)
    elif method == 'combined':
        fingerprint = gin.features.get_combined_fingerprint(smiles_str)
    else:
        raise ValueError(f"Invalid method: {method}")
    return fingerprint

# Test the function
featurize_smiles('CC(C)CC(C)(O)C1CCCS1')

# + id="construct_features"
# Construct the features `x` and labels `y` for the model
x = np.array([featurize_smiles(v) for v in data_df.index])
from sklearn.preprocessing import OrdinalEncoder
label_encoder = OrdinalEncoder()
x = label_encoder.fit_transform(x)
y = data_df[desc].values
gin.explore.pyrfume.plot_feature_heatmap(x)
save_fig(f'{desc}_feature_heatmap')

mlflow.end_run()

# + [markdown] id="604a9a65"
# Having noticed the above, we should maybe be thinking about the following:
#
# - Feature scaling - less necessary for tree-based models
# - High class cardinality - hopefully not an issue
# - Feature imbalance - this is a possible issue, but we can address this
#   - SMOTE is an option for increasing the minority class
#
# ## Splitting the data, cross-validation
# We have to split the data into training and testing sets.

# + id="split_data"
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

smote = SMOTE()
# Resampling the training data to address class imbalance
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_test_res, y_test_res = smote.fit_resample(X_test, y_test)

# + [markdown] id="6b56123a"
# ## Train and evaluate a Random Forest (RF) model
#
# We will use the RF implementation from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

# + id="train_rf"
# Start a parent run for the RandomForest models
mlflow.start_run(run_name="RandomForest_parent")
mlflow_annotate({"model_type": "RandomForest"})

# What hyper-parameters should we use?
best_params = {'bootstrap': False, 
               'max_depth': None, 
               'max_features': 'log2', 
               'min_samples_leaf': 1, 
               'min_samples_split': 5, 
               'n_estimators': 300}  # WARNING: tuned on Floral molecules -- may not apply to others

log_params({'rf_' + key: value for key, value in best_params.items()})

model = sklearn_ensemble.RandomForestClassifier(**best_params)
model_res = sklearn_ensemble.RandomForestClassifier(**best_params)

# + id="rf_model_1"
# Start a child run for the first RandomForest model
mlflow.start_run(run_name="RandomForest_model_1", nested=True)
mlflow_annotate({"model_type": "RandomForest", "model_version": "v1"})

# Fit and predict
rf_y_pred = model.fit(X_train, y_train).predict(X_test)

# Log metrics for RandomForest model_1
metrics_rf = gin.validate.get_metrics(y_test, rf_y_pred)
log_metrics(metrics_rf)

# Log confusion matrix
gin.validate.plot_confusion_matrix(y_test, rf_y_pred, suptitle="Random Forest")
save_fig(f'{desc}_confusion_matrix_rf')

# + id="rf_thresholds"
# Restart mlflow run for RandomForest threshold evaluation, same name
# Evaluate thresholds for Random Forest models
thresholds = np.arange(0, 1, 0.01)
results_df = gin.validate.evaluate_thresholds(model, 
                                              X_test, 
                                              y_test, 
                                              thresholds)
mlflow.log_params({'thresholds': thresholds})
mlflow.log_table(results_df, "threshold/threshold_results.json")

# Plot threshold results
gin.validate.plot_threshold_results(results_df, model_name='Random Forest', suptitle='Random Forest')
save_fig(f'{desc}_threshold_results_rf')

# End child run
mlflow.end_run()

# + id="rf_model_resampled"
# Start a child run for the RandomForest model with resampled data
mlflow.start_run(run_name="RandomForest_model_resampled", nested=True)
mlflow_annotate({"model_type": "RandomForest", "model_version": "v1"})

# Fit and predict
rf_y_pred_res = model_res.fit(X_train_res, y_train_res).predict(X_test_res)
rf_y_pred_res2uns = model_res.predict(X_test)

# Log metrics for RandomForest model_resampled
metrics_rf_resampled = gin.validate.get_metrics(y_test, rf_y_pred_res2uns)
log_metrics(metrics_rf_resampled)

# Log confusion matrix
gin.validate.plot_confusion_matrix(y_test, rf_y_pred_res2uns, suptitle="Random Forest - Resampled")
save_fig(f'{desc}_confusion_matrix_rf_resampled')

#+
results_df_res = gin.validate.evaluate_thresholds(model_res, X_test, y_test, thresholds)
mlflow.log_params({'thresholds': thresholds})
mlflow.log_table(results_df_res, "threshold/threshold_results_resampled.json")
gin.validate.plot_threshold_results(results_df_res, model_name='Random Forest - Resampled', suptitle='Random Forest - Resampled')
save_fig(f'{desc}_threshold_results_rf_resampled')

# End child run
mlflow.end_run()
# End parent run for RandomForest
mlflow.end_run()

# + [markdown] id="f18bca3d"
# And out of curiosity, let's also try an ensemble - even though for production-level models, this is likely overkill. A tiny performance boost often isn't worth the time and complexity.

# + id="ensemble"
# Start a parent run for Ensemble models
mlflow.start_run(run_name="Ensemble_parent")
mlflow_annotate({"model_type": "Ensemble", "model_version": "v1"})

# Define classifiers
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(max_iter=1000)
clf2 = sklearn_ensemble.RandomForestClassifier(**best_params)
clf3 = sklearn_ensemble.GradientBoostingClassifier()

# VotingClassifier with hard voting
# VotingClassifier with soft voting
model_vote = sklearn_ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')
model_vote_res = sklearn_ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')

# + id="ensemble_model_1"
# Start a child run for the ensemble model
mlflow.start_run(run_name="Ensemble_model_1", nested=True)
mlflow_annotate({"model_type": "Ensemble", "model_version": "v1"})

# Fit and predict
model_vote.fit(X_train, y_train)
eclf_y_pred = model_vote.predict(X_test)

# Log metrics
metrics_ensemble = gin.validate.get_metrics(y_test, eclf_y_pred)
log_metrics(metrics_ensemble)

# Log confusion matrix
gin.validate.plot_confusion_matrix(y_test, eclf_y_pred, suptitle="Ensemble")
save_fig(f'{desc}_confusion_matrix_ensemble')

# End child run
mlflow.end_run()

# + id="ensemble_model_resampled"
# Start a child run for the ensemble model with resampled data
mlflow.start_run(run_name="Ensemble_model_resampled", nested=True)
mlflow_annotate({"model_type": "Ensemble", "model_version": "v1"})

# Fit and predict
model_vote_res.fit(X_train_res, y_train_res)
eclf_y_pred_res2uns = model_vote_res.predict(X_test)

# Log metrics
metrics_ensemble_resampled = gin.validate.get_metrics(y_test, eclf_y_pred_res2uns)
log_metrics(metrics_ensemble_resampled)

# Log confusion matrix
gin.validate.plot_confusion_matrix(y_test, eclf_y_pred_res2uns, suptitle="Ensemble - Resampled")
save_fig(f'{desc}_confusion_matrix_ensemble_resampled')

# + id="ensemble_thresholds"
# Evaluate thresholds for Ensemble models
# Evaluate thresholds for Ensemble models
if hasattr(model_vote_res, 'predict_proba'):
    results_df_ensemble_res = gin.validate.evaluate_thresholds(model_vote_res, X_test, y_test, thresholds)
    mlflow.log_params({'thresholds': thresholds})
    mlflow.log_table(results_df_ensemble_res,
        "threshold/threshold_results_ensemble_resampled.json")

    gin.validate.plot_threshold_results(results_df_ensemble_res, model_name='Ensemble - Resampled', suptitle='Ensemble - Resampled')
    save_fig(f'{desc}_threshold_results_ensemble_resampled')
else:
    print("Warning: predict_proba is not available for this model. Skipping threshold evaluation.")

gin.validate.plot_threshold_results(results_df_ensemble_res, model_name='Ensemble - Resampled', suptitle='Ensemble - Resampled')
save_fig(f'{desc}_threshold_results_ensemble_resampled')

# End child run
mlflow.end_run()


# End parent run for Ensemble
mlflow.end_run()

# + [markdown] id="b043bdab"
# # Multi-layer Perceptron (MLP)
# Let's train a simple neural network.
#
# Now that we have tried modeling with an RF, let's try modeling with a simple neural network: the [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).

# Start a parent run for MLP models
mlflow.start_run(run_name="MLP_parent")
mlflow_annotate()

# Start a child run for MLP model
mlflow.start_run(run_name="MLP_model_1", nested=True)
mlflow_annotate({"model_type": "MLP", "model_version": "v1"})

from gin.model import MLP
model_mlp = MLP(input_dim=X_train.shape[1])
model_mlp.fit(X_train, y_train)

# Predict
mlp_y_pred = model_mlp.predict(X_test)

# Log metrics
metrics_mlp = gin.validate.get_metrics(y_test, mlp_y_pred > 0.5)
log_metrics(metrics_mlp)

# Log confusion matrix
gin.validate.plot_confusion_matrix(y_test, mlp_y_pred > 0.5, suptitle="MLP")
save_fig(f'{desc}_confusion_matrix_mlp')

# Evaluate thresholds for MLP model
results_df_mlp = gin.validate.evaluate_thresholds(model_mlp, X_test, y_test, thresholds, y_proba=mlp_y_pred)
mlflow.log_params({'thresholds': thresholds})
mlflow.log_table(results_df_mlp, "threshold/threshold_results_mlp.json")

# Plot threshold results
gin.validate.plot_threshold_results(results_df_mlp, model_name='MLP', suptitle='MLP')
save_fig(f'{desc}_threshold_results_mlp')

# End child run
mlflow.end_run()

# Start a child run for MLP model with resampled data
mlflow.start_run(run_name="MLP_model_resampled", nested=True)

model_mlp_res = MLP(input_dim=X_train_res.shape[1])
model_mlp_res.fit(X_train_res, y_train_res)

# Predict
mlp_y_pred_res = model_mlp_res.predict(X_test)

# Log metrics
metrics_mlp_resampled = gin.validate.get_metrics(y_test, mlp_y_pred_res > 0.5)
log_metrics(metrics_mlp_resampled)

# Log confusion matrix
gin.validate.plot_confusion_matrix(y_test, mlp_y_pred_res > 0.5, suptitle="MLP - Resampled")
save_fig(f'{desc}_confusion_matrix_mlp_resampled')

# Evaluate thresholds for MLP resampled model
results_df_mlp_res = gin.validate.evaluate_thresholds(model_mlp_res, X_test, y_test, thresholds, y_proba=mlp_y_pred_res)
mlflow.log_params({'thresholds': thresholds})
mlflow.log_table(results_df_mlp_res, "threshold/threshold_results_mlp_resampled.json")

# Plot threshold results
gin.validate.plot_threshold_results(results_df_mlp_res, model_name='MLP - Resampled', suptitle='MLP - Resampled')
save_fig(f'{desc}_threshold_results_mlp_resampled')

# End child run
mlflow.end_run()

# End parent run for MLP
mlflow.end_run()

# + [markdown] id="5b528b92"
# ## Conclusions
#
# The `MLP` achieves a very similar performance to the simpler methods above.
#
# Notably, the resampled model for the `MLP` does **not** perform any better, unlike the `RandomForest` above. In practice, we could try other methods of rebalancing and data augmentation techniques given sparse samples.

# + [markdown] id="bcbbe320-e334-4f9f-ab4e-bc9b72083a85"
# # Graph Neural Network (GNN), Bonus - Naive (👶)
#
# For fun, let's dovetail this section with a very naive message-passing GNN approach 🤖
#
# *NOTES OF INTEREST* 📝
# - We are doing this without resampling/data-augmentation -- so we _may not approach_ performance above.
# - Instead, using a simpler class-imbalance reweighting function in the cross-entropy objective.
# - We may not have enough samples to utilize the capacity of a bigger model.

# + id="gnn_setup"
# Start a parent run for GNN models
mlflow.start_run(run_name="GNN_parent")
mlflow_annotate({"model_type": "GNN"})

# Convert the SMILES strings to graph data and split into train/test sets
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import DataLoader
from gin.extra.features import smiles_to_graph
from gin.extra.gnn import train_gnn_model

# Convert the SMILES strings to graph data
data_list = []
for smile_string, floral in zip(data_df.index, data_df[desc]):
    data = smiles_to_graph(smile_string)
    if data is not None:
        data.y = torch.tensor([floral], dtype=torch.float)  # Assign target value
        data_list.append(data)
data_list = gin.extra.features.normalize_data_list(data_list)  # Normalize features

if len(data_list) == 0:
    raise ValueError("No valid graph data could be generated from the provided SMILES strings.")

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Train the GNN model on the training data
model_gnn = train_gnn_model(train_data, num_epochs=250)

# Run inference
model_gnn.eval()
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        preds = model_gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.extend(preds.numpy().flatten())
        all_labels.extend(batch.y.numpy().flatten())

# Plot distributions
import matplotlib.pyplot as plt

plt.hist(all_preds, bins=20, alpha=0.75, label='Predictions')
plt.hist(all_labels, bins=20, alpha=0.75, label='True Labels')
plt.legend()
plt.title('Distribution of Predictions and True Labels')
save_fig(f'{desc}_gnn_predictions_distribution')
plt.show()

# Evaluate performance
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
metrics_gnn = gin.validate.get_metrics(all_labels, all_preds > 0.5)
log_metrics(metrics_gnn)

# Log confusion matrix
gin.validate.plot_confusion_matrix(all_labels, all_preds > 0.5, suptitle='GNN Model')
save_fig(f'{desc}_confusion_matrix_gnn')

# Evaluate thresholds for GNN model
from gin.extra.validate import evaluate_thresholds_gnn
thresholds = np.arange(0.0, 1.0, 0.01)
results_gnn = evaluate_thresholds_gnn(model_gnn, test_data, thresholds)

mlflow.log_params({'thresholds': thresholds})
mlflow.log_table(results_gnn, "threshold/threshold_results_gnn.json")

# Plot threshold results
gin.validate.plot_threshold_results(results_gnn, model_name="GNN")
save_fig(f'{desc}_threshold_results_gnn')

# End parent run for GNN
mlflow.end_run()

# + [markdown] id="5b528b92"
# ## Conclusions
#
# The `GNN` model, while an interesting exercise, does not perform as well as the simpler models. This is typical for neural networks with smaller datasets.
#
# **Better yet** -- pull a model from HuggingFace 🤗 that has been pre-trained on other molecules to leverage the knowledge seen in other data.
#
# # The End
print("FINISHED!")
