import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
# from torch_scatter import scatter_mean
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(MessagePassingLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.message_function = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.update_function = nn.GRUCell(node_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        messages = torch.cat([x[row], x[col], edge_attr], dim=-1)
        messages = self.message_function(messages)
        aggregated_messages = torch.zeros_like(x)
        aggregated_messages.index_add_(0, row, messages)
        x = self.update_function(aggregated_messages, x)
        return x

class GNNModule(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, dropout_rate=0.5):
        super(GNNModule, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.message_passing_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Use sigmoid for binary classification
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embedding(x)
        for layer in self.message_passing_layers:
            x = layer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)  # Aggregating node features
        output = self.readout(x)
        return output

def calculate_class_weights(data_list):
    y = torch.cat([data.y for data in data_list], dim=0)
    class_counts = torch.bincount(y.long())
    class_weights = 1.0 / class_counts.float()
    return class_weights

def train_gnn_model(data_list, num_epochs=100, batch_size=1024, learning_rate=0.001, 
                    model=None):
    # Setup TensorBoard
    writer = SummaryWriter()

    data_loader = DataLoader(data_list, batch_size=int(3522/2), shuffle=True)
    
    # Define model
    if model is None:
        node_dim = data_list[0].x.shape[1]
        edge_dim = data_list[0].edge_attr.shape[1]
        hidden_dim = 64  # Increased hidden dimension
        num_layers = 6  # Increased number of layers
        model = GNNModule(node_dim, edge_dim, hidden_dim, num_layers)
    
    # Calculate class weights
    class_weights = calculate_class_weights(data_list)
    class_weights = class_weights / class_weights.sum()  # Normalize weights

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # Learning rate scheduler
    criterion = nn.BCELoss(reduction='none')  # Binary Cross-Entropy loss

    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        total_loss = 0.0
        current_y_values = []
        current_y_preds = []
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(output.view(-1), batch.y.view(-1))
            
            # Apply class weights
            weights = class_weights[batch.y.long()].view(-1)
            weighted_loss = (loss * weights).mean()
            
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()
            current_y_values = batch.y.numpy().flatten()
            current_y_preds = output.detach().numpy().flatten()
        
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        # Log the average loss for the epoch to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_histogram('Y/true', current_y_values, epoch)
        writer.add_histogram('Y/pred', current_y_preds, epoch)
        
        # Step the learning rate scheduler
        scheduler.step()
    
    writer.close()
    return model

if __name__ == "__main__":
    import gin
    import numpy as np
    from gin.data import read_local_csv
    from gin.extra.features import smiles_to_graph
    
    # Read the CSV data
    data_df = read_local_csv()
    
    # Convert the SMILES strings to graph data
    data_list = []
    for smile_string, floral in zip(data_df.index, data_df['floral']):
        data = smiles_to_graph(smile_string)
        if data is not None:
            data.y = torch.tensor([floral], dtype=torch.float)  # Assign target value
            data_list.append(data)

    gin.extra.features.normalize_data_list(data_list)
    
    if len(data_list) == 0:
        raise ValueError("No valid graph data could be generated from the provided SMILES strings.")
    
    # Train the GNN model
    model = train_gnn_model(data_list, num_epochs=200)  # Increased epochs
    
    # Save the trained model
    torch.save(model.state_dict(), 'gnn_model.pth')

    model.eval()
    test_loader = DataLoader(data_list, batch_size=256, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(batch.y.numpy().flatten())

    import matplotlib.pyplot as plt

    plt.close('all')
    plt.hist(all_preds, bins=20, alpha=0.75, label='Predictions')
    plt.hist(all_labels, bins=20, alpha=0.75, label='True Labels')
    plt.legend()
    plt.title('Distribution of Predictions and True Labels')
    plt.show()
    plt.savefig('predictions_vs_labels.png')


    # Evaluate performance
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    gin.validate.evaluate_model(all_labels, all_preds > 0.5)

    from gin.extra.validate import evaluate_thresholds_gnn
    thresholds = np.arange(0.0,1.0,0.01)
    results = evaluate_thresholds_gnn(model, data_list, thresholds)
    print(results)

    gin.validate.plot_threshold_results(results, model_name="GNN")
    plt.savefig('threshold_results.png')

    optimal_threshold = thresholds[np.argmax(results['F1'])]
    gin.validate.plot_confusion_matrix(all_labels, all_preds > optimal_threshold, suptitle='GNN Model')
    plt.savefig('confusion_matrix.png')
