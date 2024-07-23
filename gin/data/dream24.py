import pandas as pd
import os
# module_folder = os.path.dirname(__file__)
# print(module_folder)

_data_default =  '~/Code/repos/SMELL/DREAM/data/'

def load_dragon_descriptors():
    """
    Load Dragon Descriptors CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 'Dragon_Descriptors.csv'))

def load_mixure_definitions_test_set():
    """
    Load Mixure Definitions Test Set CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 'Mixure_Definitions_Test_set.csv'))

def load_leaderboard_set_submission_form():
    """
    Load Leaderboard Set Submission Form CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 'Leaderboard_set_Submission_form.csv'))

def load_synapse_metadata_manifest():
    """
    Load Synapse Metadata Manifest TSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 'Synapse_Metadata_Manifest.tsv'),
                       sep='\t')

def load_mixure_definitions_leaderboard_set():
    """
    Load Mixure Definitions Leaderboard Set CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 'Mixure_Definitions_Leaderboard_set.csv'))

def load_test_set_submission_form():
    """
    Load Test Set Submission Form CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 'Test_set_Submission_form.csv'))

def load_mixure_definitions_training_set():
    """
    Load Mixure Definitions Training Set CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, "Mixure_Definitions_Training_set.csv"))

def load_training_data_mixturedist():
    """
    Load Training Data Mixture Distribution CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 'TrainingData_mixturedist.csv'))


if __name__ == '__main__':

    from gin.data import *

    # Load Dragon Descriptors
    dragon_descriptors = load_dragon_descriptors()
    print(dragon_descriptors.head())

    # Load Mixure Definitions Test Set
    mixure_definitions_test_set = load_mixure_definitions_test_set()
    print(mixure_definitions_test_set.head())

    # Load Leaderboard Set Submission Form
    leaderboard_set_submission_form = load_leaderboard_set_submission_form()
    print(leaderboard_set_submission_form.head())

    # Load Synapse Metadata Manifest
    synapse_metadata_manifest = load_synapse_metadata_manifest()
    print(synapse_metadata_manifest.head())

    # Load Mixure Definitions Leaderboard Set
    mixure_definitions_leaderboard_set = load_mixure_definitions_leaderboard_set()
    print(mixure_definitions_leaderboard_set.head())

    # Load Test Set Submission Form
    test_set_submission_form = load_test_set_submission_form()
    print(test_set_submission_form.head())

    # Load Mixure Definitions Training Set
    mixure_definitions_training_set = load_mixure_definitions_training_set()
    print(mixure_definitions_training_set.head())

    # Load Training Data Mixture Distribution
    training_data_mixturedist = load_training_data_mixturedist()
    print(training_data_mixturedist.head())
