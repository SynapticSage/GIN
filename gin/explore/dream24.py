
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gin.data.dream24 import (
    load_dragon_descriptors,
    load_mixure_definitions_test_set,
    load_leaderboard_set_submission_form,
    load_synapse_metadata_manifest,
    load_mixure_definitions_leaderboard_set,
    load_test_set_submission_form,
    load_mixure_definitions_training_set,
    load_training_data_mixturedist
)

def explore_dragon_descriptors():
    df = load_dragon_descriptors()
    print("Dragon Descriptors:")
    print(df.head())
    print(df.describe())
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Dragon Descriptors')
    plt.show()

def explore_mixure_definitions_training_set():
    df = load_mixure_definitions_training_set()
    print("Mixure Definitions Training Set:")
    print(df.head())
    
    molecule_counts = df.drop(columns=['author', 'mixture_number']).sum(axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(molecule_counts, bins=20)
    plt.title('Distribution of Molecules per Mixture in Training Set')
    plt.xlabel('Number of Molecules')
    plt.ylabel('Frequency')
    plt.show()

def explore_training_data_mixturedist():
    df = load_training_data_mixturedist()
    print("Training Data Mixture Distribution:")
    print(df.head())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['distance'], bins=20, kde=True)
    plt.title('Distribution of Mixture Distances in Training Data')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()

def explore_leaderboard_set_submission_form():
    df = load_leaderboard_set_submission_form()
    print("Leaderboard Set Submission Form:")
    print(df.head())

def explore_synapse_metadata_manifest():
    df = load_synapse_metadata_manifest()
    print("Synapse Metadata Manifest:")
    print(df.head())

def explore_mixure_definitions_leaderboard_set():
    df = load_mixure_definitions_leaderboard_set()
    print("Mixure Definitions Leaderboard Set:")
    print(df.head())

def explore_test_set_submission_form():
    df = load_test_set_submission_form()
    print("Test Set Submission Form:")
    print(df.head())

def explore_mixure_definitions_test_set():
    df = load_mixure_definitions_test_set()
    print("Mixure Definitions Test Set:")
    print(df.head())

if __name__ == '__main__':
    # Explore each dataset
    explore_dragon_descriptors()
    explore_mixure_definitions_training_set()
    explore_training_data_mixturedist()
    explore_leaderboard_set_submission_form()
    explore_synapse_metadata_manifest()
    explore_mixure_definitions_leaderboard_set()
    explore_test_set_submission_form()
    explore_mixure_definitions_test_set()

