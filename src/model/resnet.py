import torch
import torch.nn as nn

class TabularResNetBlock(nn.Module):
    """
    A standard ResNet block. Input dimension always matches output dimension.
    """
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(TabularResNetBlock, self).__init__()
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.ReLU()

    def forward(self, x):
        residual = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # Skip connection: straightforward addition since dimensions match
        out = out + residual
        out = self.act2(out)
        
        return out

class TabularResNet(nn.Module):
    """
    Tabular ResNet where you easily control the uniform width and total depth.
    """
    def __init__(self, input_dim, hidden_dim, num_blocks, num_tasks, dropout_rate=0.1):
        """
        Args:
            input_dim (int): Number of input features in your dataset.
            hidden_dim (int): The width of every single block in the network.
            num_blocks (int): The total number of residual blocks to stack (depth).
            num_tasks (int): Number of regression targets to predict.
            dropout_rate (float): Dropout probability.
        """
        super(TabularResNet, self).__init__()

        self.sigma2 = nn.Parameter(torch.ones(1, num_tasks))
        
        # 1. Stem: Maps the raw input features to your chosen hidden_dim
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. ResNet Blocks: Stack exactly 'num_blocks' of them
        # Using nn.Sequential simplifies the forward pass
        self.blocks = nn.Sequential(*[
            TabularResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # 3. Output Head: Maps the final hidden representation to the target variables
        self.output_head = nn.Linear(hidden_dim, num_tasks)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        predictions = nn.functional.softplus(self.output_head(x))
        return predictions

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # --- Easily tune these parameters ---
    INPUT_FEATURES = 50   
    HIDDEN_DIM = 512      # The uniform width of the entire model
    NUM_BLOCKS = 8        # The total depth of the model (stacking 8 blocks)
    NUM_TASKS = 3         # Number of regression targets
    BATCH_SIZE = 32

    # Instantiate the model
    model = TabularResNet(
        input_dim=INPUT_FEATURES, 
        hidden_dim=HIDDEN_DIM, 
        num_blocks=NUM_BLOCKS, 
        num_tasks=NUM_TASKS,
        dropout_rate=0.2
    )

    dummy_input = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    predictions = model(dummy_input)

    print(f"Model Width (Hidden Dim): {HIDDEN_DIM}")
    print(f"Model Depth (Total Blocks): {NUM_BLOCKS}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Predictions shape: {predictions.shape}") # Expected: [32, 3]