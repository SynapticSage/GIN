import os
import pandas as pd
from pyrfume import load_data, get_data_path, list_archives
from dotenv import load_dotenv
import pyrfume

load_dotenv()
pyrfume.set_data_path(os.getenv("PYRFUME_DATA"))

REMOTE_URL = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"
LOCAL_PATH = get_data_path()

def swap_shortcut_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using shortcut names.

    Args:
        df (pd.DataFrame): DataFrame with columns to rename.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    _shortcut_names = {
        "MolecularWeight": "MW",
        "IsomericSMILES": "SMILES",
        "IUPACName": "IUPAC",
    }
    # Create the renaming dictionary with existing column names as keys
    renaming_dict = {
        col: _shortcut_names.get(col, col) for col in df.columns
    }
    df = df.rename(columns=renaming_dict)
    return df

def get_behavior_dataframe(archive_name: str,
                          descriptor: str = "",
                          remote: bool = False,
                          url: str = REMOTE_URL,
                          local_path: str = LOCAL_PATH,
                          shortcut_names=True):
    """
    Fetches a DataFrame of Stimulus IDs in the presence/absence/intensity/etc
    of a specified descriptor.

    Args:
        archive_name (str): Name of the archive in Pyrfume-Data, e.g. 'leffingwell'.
        descriptor (str, optional): Name of the descriptor, e.g. 'floral'.
        remote (bool, optional): Whether to download data from Pyrfume-Data or use local copy.
            Defaults to False.
        url (str, optional): URL of Pyrfume-Data on Github. Defaults to REMOTE_URL.
        local_path (str, optional): Local path to Pyrfume-Data. Defaults to LOCAL_PATH.

    Returns:
        pd.DataFrame: DataFrame of behavior data.
    """
    if remote:
        # Construct URL for the desired file on Github
        data_url = f"{url}/{archive_name}/behavior.csv"
        df = pd.read_csv(data_url)
    else:
        # Construct local path
        file_path = os.path.join(archive_name, "behavior.csv")
        df = load_data(file_path)
    if shortcut_names:
        df = swap_shortcut_names(df)

    return df


def get_stimuli_dataframe(archive_name: str,
                          remote: bool = False,
                          url: str = REMOTE_URL,
                          local_path: str = LOCAL_PATH,
                          shortcut_names=True):
    """
    Fetches a DataFrame of Stimulus IDs and their corresponding CIDs.

    Args:
        archive_name (str): Name of the archive in Pyrfume-Data, e.g. 'leffingwell'.
        remote (bool, optional): Whether to download data from Pyrfume-Data or use local copy.
            Defaults to False.
        url (str, optional): URL of Pyrfume-Data on Github. Defaults to REMOTE_URL.
        local_path (str, optional): Local path to Pyrfume-Data. Defaults to LOCAL_PATH.

    Returns:
        pd.DataFrame: DataFrame of stimuli data.
    """
    if remote:
        # Construct URL for the desired file on Github
        data_url = f"{url}/{archive_name}/stimuli.csv"
        df = pd.read_csv(data_url)
    else:
        # Construct local path
        file_path = os.path.join(archive_name, "stimuli.csv")
        df = load_data(file_path)
    if shortcut_names:
        df = swap_shortcut_names(df)

    return df


def get_molecules_dataframe(archive_name: str,
                            remote: bool = False,
                            url: str = REMOTE_URL,
                            local_path: str = LOCAL_PATH,
                            shortcut_names=True):
    """
    Fetches a DataFrame of Molecule information (CID, SMILES, etc.).

    Args:
        archive_name (str): Name of the archive in Pyrfume-Data, e.g. 'leffingwell'.
        remote (bool, optional): Whether to download data from Pyrfume-Data or use local copy.
            Defaults to False.
        url (str, optional): URL of Pyrfume-Data on Github. Defaults to REMOTE_URL.
        local_path (str, optional): Local path to Pyrfume-Data. Defaults to LOCAL_PATH.

    Returns:
        pd.DataFrame: DataFrame of molecule data.
    """
    if remote:
        # Construct URL for the desired file on Github
        data_url = f"{url}/{archive_name}/molecules.csv"
        df = pd.read_csv(data_url)
    else:
        # Construct local path
        file_path = os.path.join(archive_name, "molecules.csv")
        df = load_data(file_path)
    if shortcut_names:
        df = swap_shortcut_names(df)

    return df


def load(archive_name: str,
         types: list = ["behavior", "stimuli", "molecules"],
         descriptor: str = "",
         remote: bool = False,
         url: str = REMOTE_URL,
         local_path: str = LOCAL_PATH,
         shortcut_names:bool=True):
    """
    Loads data from a specified Pyrfume-Data archive.

    Args:
        archive_name (str): Name of the archive in Pyrfume-Data, e.g. 'leffingwell'.
        types (list, optional): Types of data to load. Options are 'behavior', 'stimuli',
            and 'molecules'. Defaults to ["behavior", "stimuli", "molecules"].
        descriptor (str, optional): Name of the descriptor to filter behavior data.
            Defaults to "".
        remote (bool, optional): Whether to download data from Pyrfume-Data or use local copy.
            Defaults to False.
        url (str, optional): URL of Pyrfume-Data on Github. Defaults to REMOTE_URL.
        local_path (str, optional): Local path to Pyrfume-Data. Defaults to LOCAL_PATH.

    Returns:
        dict: Dictionary containing the loaded DataFrames. Keys are the data types specified
            in 'types'.
    """
    dataframes = {}

    if "behavior" in types:
        dataframes["behavior"] = get_behavior_dataframe( archive_name,
                                                        descriptor, remote,
                                                        url, local_path,
                                                        shortcut_names)
    if "stimuli" in types:
        dataframes["stimuli"] = get_stimuli_dataframe(archive_name, remote,
                                                      url, local_path,
                                                      shortcut_names)
    if "molecules" in types:
        dataframes["molecules"] = get_molecules_dataframe(archive_name, remote,
                                                          url, local_path,
                                                          shortcut_names)

    return dataframes

def join(dataframes: dict, kind='inner', collapse_duplicates=True):
    """Join the dataframes on common indices/columns

    Args:
        dataframes (dict): DataFrames to join, keyed by data type.
        kind (str, optional): Type of join to perform. Defaults to 'inner'.

    Returns:
        pd.DataFrame: Joined dataframe.
    """
    
    df = None
    if 'behavior' in dataframes:
        df = dataframes['behavior']
    if 'stimuli' in dataframes:
        if df is None:
            df = dataframes['stimuli']
        else:
            df = df.join(dataframes['stimuli'], how=kind)
    if 'molecules' in dataframes:
        if df is None:
            df = dataframes['molecules']
        else:
            # Specify suffixes to avoid overlapping column names
            df = df.join(dataframes['molecules'], how=kind, on='CID', lsuffix='_stimuli', rsuffix='_molecules')

            if collapse_duplicates:
                columns_to_drop = []
                for col in df.columns:
                    if col.endswith("_molecules"):
                        base_col = col.replace("_molecules", "")
                        if f"{base_col}_stimuli" in df.columns:
                            # Keep the stimuli column, drop the molecules column
                            columns_to_drop.append(col)
                df = df.drop(columns=columns_to_drop)
                # Rename the stimuli columns to remove the suffix
                df.columns = [col.replace("_stimuli", "") for col in df.columns]
    return df

def get_join(archive_name: str,
             types: list = ["behavior", "stimuli", "molecules"],
             descriptor: str = "",
             remote: bool = False,
             url: str = REMOTE_URL,
             local_path: str = LOCAL_PATH,
             shortcut_names:bool=True):
    """
    Fetches and joins data from a specified Pyrfume-Data archive.

    Args:
        archive_name (str): Name of the archive in Pyrfume-Data, e.g. 'leffingwell'.
        types (list, optional): Types of data to load. Options are 'behavior', 'stimuli',
            and 'molecules'. Defaults to ["behavior", "stimuli", "molecules"].
        descriptor (str, optional): Name of the descriptor to filter behavior data.
            Defaults to "".
        remote (bool, optional): Whether to download data from Pyrfume-Data or use local copy.
            Defaults to False.
        url (str, optional): URL of Pyrfume-Data on Github. Defaults to REMOTE_URL.
        local_path (str, optional): Local path to Pyrfume-Data. Defaults to LOCAL_PATH.

    Returns:
        pd.DataFrame: Joined dataframe.
    """
    data = load(archive_name, types, descriptor, remote, url, local_path, shortcut_names)
    return join(data)

if __name__ == "__main__":
    # Demonstration
    df = get_join("leffingwell", types=["behavior", "stimuli", 'molecules'])
    print(df.head())
