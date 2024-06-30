import torch

class MLPModule(torch.nn.Module):
  def __init__(self,
               input_dim: int|np.ndarray = 2048,
               hidden_dim: int = 128):
    super().__init__()
    if isinstance(input_dim, np.ndarray):
      input_dim = input_dim.shape[-1]
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.model = self._create_model()

  def _create_model(self) -> torch.nn.Module:
    """Create a three layer MLP producing a prediction tensor [n_samples, 1]."""
    return torch.nn.Sequential(
        torch.nn.Linear(self.input_dim, self.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(self.hidden_dim, 1),
        torch.nn.Sigmoid()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.squeeze(self.model(x), dim=-1)

class MLP:
  def __init__(self, input_dim: int = 2048):
    self.net = MLPModule(input_dim)
    self.optimizer = self._get_optimizer()
    self.criterion = self._get_criterion()

  def _get_optimizer(self) -> torch.optim.Optimizer:
    """Setup a PyTorch optimizer."""
    return torch.optim.Adam(self.net.parameters(), lr=1e-3)

  def _get_criterion(self) -> torch.nn.Module:
    """Setup a loss function."""
    return torch.nn.BCELoss()

  def _train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Train one epoch with data loader and return the average train loss.

      Please make use of the following initialized items:
      - self.net
      - self.optimizer
      - self.criterion

    """
    self.net.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        self.optimizer.zero_grad()
        predictions = self.net(batch_x)
        loss = self.criterion(predictions, batch_y)
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

  def fit(self, x: np.ndarray, y: np.ndarray, num_epoch: int = 200, batch_size: int = 256):
    from tqdm.auto import tqdm
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for current_epoch in tqdm(range(num_epoch), desc='Epochs', total=num_epoch):
      loss = self._train_one_epoch(dataloader)
      # if current_epoch % 10 == 0:
        # print(f'Epoch {current_epoch}:\t loss={loss:.4f}')

  def predict(self, x: np.ndarray):
    self.net.eval()
    return self.net(torch.Tensor(x)).detach().cpu().numpy()

