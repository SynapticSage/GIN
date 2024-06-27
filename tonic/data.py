import os
import pandas as pd
import requests

DATA_URL = 'https://raw.githubusercontent.com/pyrfume/pyrfume-data/af976163b76931363c74c7da5e08469f42217966/leffingwell/behavior.csv'
LOCAL_PATH = 'data/behavior.csv'

def download_local_csv(url: str=DATA_URL, 
                       local_path: str=LOCAL_PATH):

    """Download the CSV from a URL to a local file."""

    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded data to {local_path}")

def read_local_csv(local_path: str=LOCAL_PATH) -> pd.DataFrame:

    """Read the local CSV into a DataFrame."""
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"{local_path} does not exist")
    return pd.read_csv(local_path).set_index('IsomericSMILES')[['floral']]

def read_remote_csv(url: str=DATA_URL) -> pd.DataFrame:

    """Read the remote CSV into a DataFrame."""

    return pd.read_csv(url).set_index('IsomericSMILES')[['floral']]

if __name__ == "__main__":
    from tonic.data import read_local_csv
    from tonic.extra.features import smiles_to_graph
    
    # Read the CSV data
    data_df = read_local_csv()
    
    # Convert the SMILES strings to graph data
    data_list = []
    for smile_string, floral in zip(data_df.index, data_df['floral']):
        data = smiles_to_graph(smile_string)
        if data is not None:
            data.y = torch.tensor([floral], dtype=torch.float)  # Assign target value
            data_list.append(data)
    
    # Train the GNN model
    model = train_gnn_model(data_list)
    
    # Save the trained model
    torch.save(model.state_dict(), 'gnn_model.pth')
