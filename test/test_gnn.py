import pytest
import torch
from torch_geometric.data import Data
from tonic.extra.features import smiles_to_graph
from tonic.extra.gnn import GNNModule, MessagePassingLayer, train_gnn_model

@pytest.fixture
def example_data():
    # Create example data for testing
    smiles = "CCO"  # Ethanol
    data = smiles_to_graph(smiles)
    data.y = torch.tensor([1.0], dtype=torch.float)  # Assume target is 1.0 (floral)
    return data

def test_smiles_to_graph():
    smiles = "CCO"  # Ethanol
    data = smiles_to_graph(smiles)
    assert data.x.shape[0] == 3  # 3 atoms
    assert data.edge_index.shape[1] == 4  # 2 bonds, bidirectional
    assert data.edge_attr.shape[0] == 4  # 2 bonds, bidirectional

def test_message_passing_layer(example_data):
    node_dim = example_data.x.shape[1]
    edge_dim = example_data.edge_attr.shape[1]
    layer = MessagePassingLayer(node_dim, edge_dim)
    output = layer(example_data.x, example_data.edge_index, example_data.edge_attr)
    assert output.shape == example_data.x.shape

def test_gnn_module(example_data):
    node_dim = example_data.x.shape[1]
    edge_dim = example_data.edge_attr.shape[1]
    hidden_dim = 128
    num_layers = 2
    model = GNNModule(node_dim, edge_dim, hidden_dim, num_layers)
    output = model(example_data.x, example_data.edge_index, example_data.edge_attr, torch.zeros(example_data.x.shape[0], dtype=torch.long))
    assert output.shape == torch.Size([1, 1])

def test_train_gnn_model(example_data):
    # Duplicate the example data to create a small dataset
    data_list = [example_data for _ in range(10)]
    model = train_gnn_model(data_list, num_epochs=10, batch_size=2, use_smote=False)
    assert model is not None
    assert isinstance(model, GNNModule)

if __name__ == "__main__":
    pytest.main()

