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
# The following cell clones a private GitHub repo called `gin`
#  -- where I developed the code for this analysis.
#
# Even with notebooks, I tend to modularize pieces into repos for
# - reproducibility
# - CI/CD
# - testing
#
# The "fine-grained" `access_token` token below grants permission to pull the private repo.
# -



# + colab={"base_uri": "https://localhost:8080/"} id="030e98c5-f7f4-4811-98a0-fc97b9cc0ce3" outputId="c5032aab-8eeb-422a-e2cf-f23660c85260"
import importlib
import os
import shutil

# Check if the `gin` package is installed
module_spec = importlib.util.find_spec('gin')
module_spec

# Check if the `gin` package is installed, if refresh is True, then we will refresh the package
refresh = False
if module_spec and refresh:
    shutil.rmtree(folder)
    gin_path = os.path.dirname(module_spec.locations)
    # NOTE: This is an access token fenced-off for this specific private repository - only usable to clone this single private repo.
    repo_url = f'https://github.com/synapticsage/gin.git'
    os.system(f'git clone {repo_url}')
    os.chdir('gin')
    # # !pip install . 
    # pip install the package
    # os.chdir('..')

# + id="fa70a528-888d-41b6-af4a-b8f0b1f2206e"
# %matplotlib inline

import gin

import numpy as np
import torch
from sklearn import ensemble as sklearn_ensemble
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
# import seaborn as sns
plt.rcParams['figure.dpi'] = 150

# + [markdown] id="d538e6fc"
# # Dataset
#
# Here we will use data managed by [the Pyrfume project](https://pyrfume.org/) 
#
# The [SMILES strings](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) representing the molecular structures and their corresponding binary labels are provided.

# + id="264e66a1"
data_df = gin.data.pyrfume.read_local_csv()

# + colab={"base_uri": "https://localhost:8080/", "height": 455} id="314b5dc3-e331-4555-8184-739d0b2f684f" outputId="d64681f6-2985-40f1-f2b0-5902d747b84f"
data_df

# + [markdown] id="af266b7c"
#
# Now that we have the data loaded, what should we learn about this dataset?

# + colab={"base_uri": "https://localhost:8080/"} id="9e91f4ee" outputId="676dd13a-287f-4809-8f5e-4e624d7baa34"
data_df['floral'].isnull().sum()

# + [markdown] id="2fd33ec1"
#
# No missing values - reassuring!
#
# Let's see the distribution of the labels in the dataset.

# + colab={"base_uri": "https://localhost:8080/", "height": 546} id="714f0c21" outputId="93543d52-12f6-4dfb-a811-ee1232cb13a1"
gin.explore.pyrfume.plot_floral_distribution(data_df, kind='pie')

# + [markdown] id="d9d9222f"
# 👆 The large majority of the dataset is non-floral ❌💐. We should consider **class imbalance** downstream.

# + colab={"base_uri": "https://localhost:8080/", "height": 576} id="937701be" outputId="f85045b4-86c4-4187-89cf-ebbfa0a6d148"
# Let's visualize some of the molecular structures in the dataset and see if we can spot any patterns.
gin.explore.pyrfume.plot_molecular_structures_w_label(data_df, num_samples=20)

# + [markdown] id="2059e7bf"
# And let's examine a few more samples.

# + colab={"base_uri": "https://localhost:8080/", "height": 576} id="1caefc10" outputId="cb27d07c-46e1-45c9-c6b4-73a9c855bf77"
gin.explore.pyrfume.plot_molecular_structures_w_label(data_df, num_samples=20)

# + [markdown] id="3b8a8363"
# ## Hypotheses
#
# Some things of note just from the visualization (hypotheses / possibillium / wild guesses):
#
# - The 🪻 floral molecules nearly all have oxygen with free electron pair. Doubled-bond oxygen alone  seems less often associated with floral molecules.
# - Nitrogen-containing rarely floral - though, devil's advocate, I also see fewer nitrogen-containing molecules to form an opinion.
#
# # Molecule Featurization
#
# In the next step, we will try to "digitize" each molecule by creating a 1D numpy array based on its molecular structure. Can you create a molecular fingerprint with `rdkit` ([documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity))?

# + colab={"base_uri": "https://localhost:8080/"} id="8133b2dc" outputId="7e3f1d34-986d-4a62-877d-2211f6cdd959"
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

featurize_smiles('CC(C)CC(C)(O)C1CCCS1')

# + colab={"base_uri": "https://localhost:8080/", "height": 392} id="6c58eeaa" outputId="8fb58cc9-43f7-4c79-efab-250dcb1f04b6"
# Construct the features `x` and labels `y` for the model
x = np.array([featurize_smiles(v) for v in data_df.index])
from sklearn.preprocessing import OrdinalEncoder
label_encoder = OrdinalEncoder()
x = label_encoder.fit_transform(x)
y = data_df['floral'].values
gin.explore.pyrfume.plot_feature_heatmap(x)

# + [markdown] id="604a9a65"
# Having noticed the above, we should maybe be thinking about the following
#
# - Feature scaling - less necessary for tree-based models
# - High class cardinality - hopefully not an issue
# - Feature imbalance - this is a possible issue, but we can address this
#   - SMOTE is an option for increasing the minority class
#
# ## Splitting the data, cross-validation
# we have to split the data into training and testing sets.

# + id="72be4cf4"
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

smote = SMOTE()
# Resampling before splitting the data can lead to data leakage
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_test_res, y_test_res = smote.fit_resample(X_test, y_test)

# + [markdown] id="6b56123a"
# ## Train and evaluate a random forest (RF) model
#
# We will use the RF implementation from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

# + id="6b56123a"
# What hyper-parameter should we use?
best_params = {'bootstrap': False, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}

model = sklearn_ensemble.RandomForestClassifier(**best_params)
model_res = sklearn_ensemble.RandomForestClassifier(**best_params)

# How do we fit and inference with the model?
rf_y_pred = model.fit(X_train, y_train).predict(X_test)
rf_y_pred_res = model_res.fit(X_train_res, y_train_res).predict(X_test_res)
rf_y_pred_res2uns = model_res.predict(X_test)

# + [markdown] id="f18bca3d"
# And out of curiosity, let's also try an ensemble - even though for production-level models, this is likely overkill. A tiny performance boost often isn't worth the time and complexity.

# + [markdown] id="56d7f4dd"
# ## Scoring / Evaluation 📝
#
# How do we evaluate the model performance? What metrics are relevant here?
#
# This is binary classification - we care about precision, recall, F1, and AUC-ROC.

# + colab={"base_uri": "https://localhost:8080/", "height": 766} id="19f90c57" outputId="addce402-1aed-4959-96de-9632e7cb2a35"
# What sort of visualization is needed here?
print("----------------")
print("Random Forest")
print("----------------")
suptitle = 'Random Forest'
gin.validate.evaluate_model(y_test, rf_y_pred)
gin.validate.plot_confusion_matrix(y_test, rf_y_pred, suptitle=suptitle)

# + colab={"base_uri": "https://localhost:8080/", "height": 766} id="1f576b60" outputId="245b2701-1c5f-4618-8969-6ec731bc10d1"
print("----------------")
print("Random Forest - Resampled")
print("----------------")
suptitle = 'Random Forest - Resampled'
gin.validate.evaluate_model(y_test, rf_y_pred_res2uns)
gin.validate.plot_confusion_matrix(y_test, rf_y_pred_res2uns, suptitle=suptitle)

# + colab={"base_uri": "https://localhost:8080/"} id="a4f07b95" outputId="e4411c03-a672-463b-f072-9152bb24142a"
start_time = time.time()
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(max_iter=1000)
clf2 = sklearn_ensemble.RandomForestClassifier(**best_params)
clf3 = sklearn_ensemble.GradientBoostingClassifier()

# VotingClassifier with hard voting
model_vote = sklearn_ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='hard')
model_vote_res = sklearn_ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='hard')

# Fit and predict
print("Fitting vote model")
model_vote.fit(X_train, y_train)
eclf_y_pred = model_vote.predict(X_test)

print("Fitting the res vote model")
model_vote_res.fit(X_train_res, y_train_res)
eclf_y_pred_res = model_vote_res.predict(X_test_res)
eclf_y_pred_res2uns = model_vote_res.predict(X_test)

print("Time taken: ", time.time() - start_time)

# + colab={"base_uri": "https://localhost:8080/", "height": 766} id="793c8713" outputId="66382dad-b565-4fb2-8b58-d6ae96a957e2"
print("----------------")
print("Ensemble")
print("----------------")
suptitle = "Ensemble"
gin.validate.evaluate_model(y_test, eclf_y_pred)
gin.validate.plot_confusion_matrix(y_test, eclf_y_pred, suptitle=suptitle)

# + colab={"base_uri": "https://localhost:8080/", "height": 766} id="c3c0307b" outputId="010dd927-72a9-4477-924e-1499e6556a8a"
print("----------------")
print("Ensemble - Resampled")
print("----------------")
suptitle = "Ensemble - Resampled"
gin.validate.evaluate_model(y_test, eclf_y_pred_res2uns)
gin.validate.plot_confusion_matrix(y_test, eclf_y_pred_res2uns, suptitle=suptitle)

# + [markdown] id="4c7cf486"
# By default random forest sets a default, but perhaps that's not ideal. We have a great deal of choice for type I type II error, and situationally these change.
#
# So let's examine how everything changes as a function of threshold.
#
# > Note: for hyperparameter tuning, usually we want a train, test, and validation set. But since it already works well above with SMOTE, I'm going to forgo a validation set for this exercise 😈.

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="fc517317" outputId="f41f7a59-f971-4668-9470-862402eb6521"
thresholds = np.arange(0, 1, 0.01)
results_df_res = gin.validate.evaluate_thresholds(model_res, X_test,
                                                  y_test, thresholds)
results_df = gin.validate.evaluate_thresholds(model, X_test, y_test,
                                              thresholds)
gin.validate.plot_threshold_results(results_df_res, model_name='Random Forest', suptitle='Random Forest - Resampled')
gin.validate.plot_threshold_results(results_df, model_name='Random Forest', suptitle='Random Forest')

# + [markdown] id="d1a0230f"
# ## Conclusion
#
# Class imbalance correction creates a modest improvement.
#
# The correct threshold depends on what we're optimizing for: do we want to balance precision recall for floral molecules or non-floral?
#
# Generally, we should pick a threshold somewhere in the goldilocks zone (shown in gray above).

# + [markdown] id="b043bdab"
# # Multi-layer Perceptron 
# Let's traina simple neural network.
#
# Now that we have tried modeling with an RF, let's try modeling with a simple neural network: the [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).
#
# This exercise aims to see whether we can train a PyTorch neural network from end to end, so a simple sanity check is adequate, and a thorough evaluation is **not** required.
#
# ## Build the MLP module and model API

# + [markdown] id="7af0283b"
# ## Setup a simple data loader and train the model

# + colab={"base_uri": "https://localhost:8080/", "referenced_widgets": ["6d0890f44e814206bd30c60201f45332", "a0475cc35b8b407e8a942e3fbd35cfc4", "edc52d907cda4314b9ac9460a872695d", "1f4281bceda244acb2a23c4cea260031", "84eee56305d74f52ad64ec48c5069cc6", "09f05218359641219abbc8361ccbddb0", "990c50f9973746638302901163199cba", "87ea4a0a693248218bce9a33d9030e5c", "688738e111ea43cb9cc0204e2a507b5f", "cd518f62f2764893b3c59b5ac4f69956", "3f5e690936784b86909fd851156c96fc", "788cc0bc124d42aeb7edf793a1f89673", "c31cb31f9d5e4a80a911594fa205eeb4", "0c1522e6c36041dda0bf29797885828e", "66d15d87fd9940af84329d9aef29b163", "7302c3f754f145e5bb5769198f776b29", "c1385ef173004e369c46d1e547e92cf1", "7f287cf51cc24b60ac6aca67bd509490", "b1bfc77433c044cd879ec167309e8d3c", "9d21d4e393284d6aa0cc80d4b0f25bbf", "5dae1ccc76fb4370976e37a003a42b9b", "d7d18611bb8c4eda89b9e8925d302575"]} id="71afc38b" outputId="dfcb4c9c-ea77-4064-a789-f98d40e8a569"
from gin.model import MLP
model = MLP(input_dim=X_train.shape[1])
model.fit(X_train, y_train)

model_res = MLP(input_dim=X_train_res.shape[1])
model_res.fit(X_train_res, y_train_res)

# + colab={"base_uri": "https://localhost:8080/", "height": 549} id="7c731a83" outputId="8ddff76c-2251-4e0d-f0ac-53ad9c155847"
# Sanity check — how do we know the model has learned from the data?
mlp_y_pred = model.predict(X_test)
mlp_y_pred_res = model_res.predict(X_test)
mlp_y_pred_res2uns = model_res.predict(X_test)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].hist([mlp_y_pred, y_test], bins=20, label=['MLP', 'True'], color=['gray', 'red'])
axs[1].hist([mlp_y_pred_res2uns, y_test], bins=20, label=['MLP_res', 'True'], color=['black', 'red'])
axs[0].set_title('MLP')
axs[1].set_title('MLP - Resampled')
axs[0].legend()
axs[1].legend()
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="48364b32" outputId="e7ed5066-abee-4991-e3f0-15f7f42d9216"
# As before, let's just explore the default threshold > 0.5
print("----------------")
print("MLP")
print("----------------")
suptitle = 'MLP'
gin.validate.evaluate_model(y_test, mlp_y_pred>0.5)
gin.validate.plot_confusion_matrix(y_test, mlp_y_pred>0.5, suptitle=suptitle)

print("----------------")
print("MLP - Resampled")
print("----------------")
suptitle = 'MLP - Resampled'
gin.validate.evaluate_model(y_test, mlp_y_pred_res2uns > 0.5)
gin.validate.plot_confusion_matrix(y_test, mlp_y_pred_res2uns>0.5, suptitle=suptitle)

# + [markdown] id="57d225e2"
# Let's also try to examine the threshold for the MLP's final sigmoid output.

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="5fd0516a" outputId="4bdd2001-04c9-4dd6-809f-434ff53b4712"
results_df_mlp = gin.validate.evaluate_thresholds(model,
                                                  X_test,
                                                  y_test,
                                                  thresholds,
                                                  y_proba=mlp_y_pred)
results_df_mlp_res = gin.validate.evaluate_thresholds(model_res,
                                                      X_test,
                                                      y_test,
                                                      thresholds,
                                                      y_proba=mlp_y_pred_res)
gin.validate.plot_threshold_results(results_df_mlp, model_name='MLP', suptitle='MLP')
gin.validate.plot_threshold_results(results_df_mlp_res, model_name='MLP - Resampled', suptitle='MLP - Resampled')

# + [markdown] id="5b528b92"
# ## Conclusions
#
# The `MLP` achieves a very similar performance to the simpler method above.
#
# Notably, the resampled model for the `MLP` does **not** perform any better, unlike the `RandomForest` above. In practice, we could try other methods of rebalancing and data augmentation techniques given sparse samples.
#
# # Graph Neural Network, Bonus - Naive (👶) 
#
# For fun, let's dovetail this section with a very naive message-passing GNN approach  🤖
#
# *NOTES OF INTEREST*  📝
# - We are doing this without resampling/data-augmentation -- so we _may not approach_ performance above.
# - Instead, using a simpler class-imbalance reweighting function in the cross-entropy objective.
# - We may not have enough samples to utilize the capacity of a bigger model.

# + id="bcbbe320-e334-4f9f-ab4e-bc9b72083a85"
# Convert the SMILES strings to graph data and split into train/test sets
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import DataLoader
from gin.extra.features import smiles_to_graph
from gin.extra.gnn import train_gnn_model

# Convert the SMILES strings to graph data
data_list = []
for smile_string, floral in zip(data_df.index, data_df['floral']):
    data = smiles_to_graph(smile_string)
    if data is not None:
        data.y = torch.tensor([floral], dtype=torch.float)  # Assign target value
        data_list.append(data)
data_list = gin.extra.features.normalize_data_list(data_list) # Normalize features

if len(data_list) == 0:
    raise ValueError("No valid graph data could be generated from the provided SMILES strings.")

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# + [markdown] id="825370e7-e109-44fa-91bf-88479e6851b8"
# Train

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["40110c9b8b74451d8e71aea7ab593d68", "1da1d2bf253643c3adc7c3f307ce6516", "0c038ad3946c4254a9849c0d41430801", "9bb459bbae5b48288af1603c60f3d4e2", "5a119996bed041f3b3edba02b2016340", "ee218f9cc4274adb9f84bae90aba41c7", "d35859f8ea354e52a342313677684c77", "d38a773fe6a1465dae99c7ab0504682a", "cc7d67f462a245c499cc32d0e080ceec", "81b52f2e4814455fb4f11692f7c00bd3", "fffcb0fffede45e4a266fdbae08f2e73"]} id="ae92f0d1-1166-4574-b408-4afa6017f33b" outputId="47abbceb-7798-4158-ac1f-7019e27767b1"
# Train the GNN model on the training data with SMOTE applied
model = train_gnn_model(train_data, num_epochs=250)

# + [markdown] id="3dc9096a-f432-4311-a90a-4ba1440de8b2"
# And now, let's run inference

# + colab={"base_uri": "https://localhost:8080/", "height": 457} id="33fd81bc-84d1-4849-b737-d24675067ca4" outputId="fba829b2-280c-4005-abbb-0326f0170e39"
model.eval()
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.extend(preds.numpy().flatten())
        all_labels.extend(batch.y.numpy().flatten())

import matplotlib.pyplot as plt

plt.hist(all_preds, bins=20, alpha=0.75, label='Predictions')
plt.hist(all_labels, bins=20, alpha=0.75, label='True Labels')
plt.legend()
plt.title('Distribution of Predictions and True Labels')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 714} id="f7d2c730-1355-42af-bd73-0e2391148b73" outputId="68db1a35-4de8-450c-f31d-ca561d93dd07"
# Evaluate performance
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
gin.validate.evaluate_model(all_labels, all_preds > 0.5)
gin.validate.plot_confusion_matrix(all_labels, all_preds > 0.5, suptitle='GNN Model')

# + colab={"base_uri": "https://localhost:8080/", "height": 895} id="a9e26046-26f5-4c6d-889f-43b8f8615b28" outputId="52a299a7-e1fd-4319-cb29-cc77f5316920"
from gin.extra.validate import evaluate_thresholds_gnn
thresholds = np.arange(0.0,1.0,0.01)
results = evaluate_thresholds_gnn(model, test_data, thresholds)
results

# + colab={"base_uri": "https://localhost:8080/", "height": 573} id="e059722a-11e8-4210-9b03-3f9d232d2c5f" outputId="5bb94d78-67b2-45e8-d85e-4dca2b573fbb"
gin.validate.plot_threshold_results(results, model_name="GNN")

# + [markdown] id="5b528b92"
# ## Conclusions
#
# The `MLP` achieves a very similar performance to the simpler method above.
#
# Notably, the resampled model for the `MLP` does **not** perform any better, unlike the `RandomForest` above. In practice, we could try other methods of rebalancing and data augmentation techniques given sparse samples.
#
# The GNN model, while an interesting exercise, does not perform as well as the simpler models. This is typical for neural networks with smaller datasets.
#
# <h4> Better yet -- pull a model from huggingface that has been pre-trained on other molecules to leverage the knowledge seen in other data.</h4>
#
# # The End
