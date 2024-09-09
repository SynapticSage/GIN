import pandas as pd
import os
import gin
# module_folder = os.path.dirname(__file__)
# print(module_folder)

# _data_default =  '~/Code/repos/SMELL/DREAM/data/'
_data_default = os.path.abspath(os.path.join(*gin.__path__, "..", "data"))

def load_dragon_descriptors():
    """
    Load Dragon Descriptors CSV file: contains
    featues of molecules
    """
    return pd.read_csv(os.path.join(_data_default, 
                                    'Dragon_Descriptors.csv'))

def load_mixure_definitions_test_set():
    """
    Load Mixure Definitions Test Set CSV file.
    This defines a set of chemicals in a mixture.
    """
    return pd.read_csv(os.path.join(_data_default, 
                                    'Mixure_Definitions_Test_set.csv'))

def load_synapse_metadata_manifest():
    """
    Load Synapse Metadata Manifest TSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 
                                    'Synapse_Metadata_Manifest.tsv'),
                       sep='\t')

def load_test_set_submission_form():
    """
    Load Test Set Submission Form CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 
                                    'Test_set_Submission_form.csv'))

def load_mixure_definitions_training_set():
    """
    Load Mixure Definitions Training Set CSV file.

    These define the chemical components inside
    a given mixture.
    """
    return pd.read_csv(os.path.join(_data_default, 
                                    "Mixure_Definitions_Training_set.csv"))

def load_training_data_mixturedist():
    """
    Load Training Data Mixture Distribution CSV file.

    This defines the distance between two
    given mixtures measured perceptually.
    """
    return pd.read_csv(os.path.join(_data_default, 'TrainingData_mixturedist.csv'))

def load_leaderboard_set_submission_form():
    """
    Load Leaderboard Set Submission Form CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 
                                    'Leaderboard_set_Submission_form.csv'))

def load_mixure_definitions_leaderboard_set():
    """
    Load Mixure Definitions Leaderboard Set CSV file.
    """
    return pd.read_csv(os.path.join(_data_default, 
                                    'Mixure_Definitions_Leaderboard_set.csv'))


if __name__ == '__main__':
    from gin.data import *

    # Load all datasets
    d = datasets = {
        'synapse_metadata': load_synapse_metadata_manifest(),
        'dragon_descriptors': load_dragon_descriptors(),
        'training_mixture_defs': load_mixure_definitions_training_set(),
        'training_mixture_dists': load_training_data_mixturedist(),
        'test_mixture_defs': load_mixure_definitions_test_set(),
        'test_submissions_dists': load_test_set_submission_form(),
        'leaderboard_submissions': load_leaderboard_set_submission_form(),
        'leaderboard_mixture_defs': load_mixure_definitions_leaderboard_set()
    }

    def join_datasets(dataframes, kind='inner'):
        """Join the dataframes on common indices/columns"""
        df = None
        for name, dataframe in dataframes.items():
            if df is None:
                df = dataframe
            else:
                df = df.merge(dataframe, how=kind, left_index=True, right_index=True, suffixes=(f'_{name}', ''))
        return df

    # Print head of each dataset
    for name, df in datasets.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(df.head())

    merged_data_1 = pd.merge(d['training_mixture_defs'], 
                             d['dragon_descriptors'], on='CID', how='left')

    # Map mixture labels to corresponding CIDs
    # mix_1 = datasets['training_mixture_dists']['Mixture 1'].map(merged_data_1.set_index('Mixture Label')['CID'])
    # mix_2 = datasets['training_mixture_dists']['Mixture 2'].map(merged_data_1.set_index('Mixture Label')['CID'])
    #
    # # Merge with distances
    # merged_data_2 = datasets['training_mixture_dists'].copy()
    # merged_data_2['Mixture 1 CID'] = mix_1
    # merged_data_2['Mixture 2 CID'] = mix_2
    #
    # final_merged_data = pd.merge(merged_data_2, merged_data_1, left_on='Mixture 1 CID', right_on='CID', how='left')
    # final_merged_data = pd.merge(final_merged_data, merged_data_1, left_on='Mixture 2 CID', right_on='CID', how='left', suffixes=('_1', '_2'))

    """
    _defs define the chemical components inside a given mixture, where each cid in a row is a component
          sorted by highest to lowest concentration
    _dists define the distance between two given mixtures measured perceptually
    _dragon_descriptors define the features of molecules
    """
