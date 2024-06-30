from rdkit import Chem
import torch
from torch_geometric.data import Data

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetHybridization().real,
        atom.GetNumRadicalElectrons(),
        atom.GetIsAromatic(),
        atom.IsInRing(),
        atom.GetMass(),
    ]

def bond_features(bond):
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing(),
        bond.GetStereo(),
        bond.GetIsAromatic(),
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get node features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))

    # Get edge indices and edge attributes
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_features(bond))
        edge_attr.append(bond_features(bond))

    # Convert to tensors
    x = torch.tensor(atom_features_list, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def normalize_data_list(data_list):
    # Concatenate all node and edge features to compute mean and std
    all_node_features = torch.cat([data.x for data in data_list], dim=0)
    all_edge_features = torch.cat([data.edge_attr for data in data_list], dim=0)
    
    # Compute mean and std for node features
    node_mean = all_node_features.mean(dim=0)
    node_std = all_node_features.std(dim=0)
    
    # Compute mean and std for edge features
    edge_mean = all_edge_features.mean(dim=0)
    edge_std = all_edge_features.std(dim=0)
    
    # Normalize each data object
    for data in data_list:
        data.x = (data.x - node_mean) / (node_std + 1e-6)  # Adding epsilon to avoid division by zero
        data.edge_attr = (data.edge_attr - edge_mean) / (edge_std + 1e-6)
    
    return data_list

# Example usage
data_list = [Data(x=torch.rand((5, 9)), edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.rand((2, 5))) for _ in range(10)]
normalized_data_list = normalize_data_list(data_list)


if __name__ == "__main__":
    # Test the function
    smiles = "CCO"
    data = smiles_to_graph(smiles)
    print(data.x)
    print(data.edge_index)
    print(data.edge_attr)
