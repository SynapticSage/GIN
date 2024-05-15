import pandas as pd

def load_dragon_descriptors(filepath='Dragon_Descriptors.csv'):
    """
    Load Dragon Descriptors CSV file.
    """
    return pd.read_csv(filepath)

def load_mixure_definitions_test_set(filepath='Mixure_Definitions_test_set.csv'):
    """
    Load Mixure Definitions Test Set CSV file.
    """
    return pd.read_csv(filepath)

def load_leaderboard_set_submission_form(filepath='Leaderboard_set_Submission_form.csv'):
    """
    Load Leaderboard Set Submission Form CSV file.
    """
    return pd.read_csv(filepath)

def load_synapse_metadata_manifest(filepath='SYNAPSE_METADATA_MANIFEST.tsv'):
    """
    Load Synapse Metadata Manifest TSV file.
    """
    return pd.read_csv(filepath, sep='\t')

def load_mixure_definitions_leaderboard_set(filepath='Mixure_Definitions_Leaderboard_set.csv'):
    """
    Load Mixure Definitions Leaderboard Set CSV file.
    """
    return pd.read_csv(filepath)

def load_test_set_submission_form(filepath='Test_set_Submission_form.csv'):
    """
    Load Test Set Submission Form CSV file.
    """
    return pd.read_csv(filepath)

def load_mixure_definitions_training_set(filepath='Mixure_Definitions_Training_set.csv'):
    """
    Load Mixure Definitions Training Set CSV file.
    """
    return pd.read_csv(filepath)

def load_training_data_mixturedist(filepath='TrainingData_mixturedist.csv'):
    """
    Load Training Data Mixture Distribution CSV file.
    """
    return pd.read_csv(filepath)

