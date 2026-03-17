import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MLP_Embedding(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Embedding, self).__init__()
        
        layers = []
        
        # Main case: input dimension is larger than 32
        if input_dim > 32:
            # Find the largest power of 2 <= input_dim
            first_hidden_dim = 1 << (input_dim.bit_length() - 1)
            
            layers.append(nn.Linear(input_dim, first_hidden_dim))
            layers.append(nn.ReLU())
            
            # Dynamically add shrinking layers until we reach 32
            current_dim = first_hidden_dim
            while current_dim > 32:
                next_dim = current_dim // 2
                if next_dim < 32:
                    next_dim = 32
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(nn.ReLU())
                current_dim = next_dim
            
            layers.append(nn.Linear(32, 16))

        # Edge case: input dimension is 32 or smaller
        else:
            # New Rule: Use a single linear layer to map directly to 16
            layers.append(nn.Linear(input_dim, 16))
            
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        embedding = self.layers(x)
        return  self.norm(embedding)

class MLP_Embedding_old(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Embedding, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        embedding = self.layers(x)
        return  self.norm(embedding)

class ProjectionLayer(nn.Module):
    def __init__(self, output_dim, embedding_dim, bias=True):
        super(ProjectionLayer, self).__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.randn(output_dim, embedding_dim))

        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Shape: (batch_size, 49, 16) * (1, 49, 16)
        elementwise_product = x * self.weight.unsqueeze(0)
        
        # Shape: (batch_size, 49)
        output = torch.sum(elementwise_product, dim=2)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
        
class DLDSC(nn.Module):
    def __init__(self, feature_splits, output_dim, no_mix_weights=False):
        super(DLDSC, self).__init__()
            
        self.feature_splits = feature_splits
        self.output_dim = output_dim
        self.num_mlps = len(feature_splits)
        self.embedding_dim = 16
        
        self.mlps = nn.ModuleList([MLP_Embedding(len(s)) for s in self.feature_splits])
        
        # Shape: (output_dim, num_mlps)
        if no_mix_weights:
            self.mixture_logits = nn.Parameter(torch.ones(self.output_dim, self.num_mlps))
            self.mixture_logits.requires_grad = False
        else:
            self.mixture_logits = nn.Parameter(torch.randn(self.output_dim, self.num_mlps))
        
        self.projection_layer = ProjectionLayer(self.output_dim, self.embedding_dim)

        self.sigma2 = nn.Parameter(torch.ones(1, output_dim))
    
    def get_mixture_props(self):
        return self.mixture_logits

    def forward(self, x):
        # Shape: (output_dim, num_mlps)
        mixture_weights = self.mixture_logits
        
        # Shape (each element): (batch_size, embedding_dim)
        split_x = [x[:,s] for s in self.feature_splits]

        # Shape: (batch_size, embedding_dim, mlps)
        mlp_embeddings = torch.stack([self.mlps[i](split_x[i]) for i in range(self.num_mlps)], dim=2)

        # mlp_embeddings reshaped: (batch_size, 1, embedding_dim, num_mlps)
        # mixture_weights reshaped: (1, output_dim, 1, num_mlps)
        # Shape: (batch_size, output_dim, embedding_dim, num_mlps)
        weighted_embeddings = mlp_embeddings.unsqueeze(1) * mixture_weights.unsqueeze(0).unsqueeze(2)
        
        # Shape: (batch_size, output_dim, embedding_dim)
        merged_embeddings = torch.sum(weighted_embeddings, dim=3)

         # Shape: (batch_size, output_dim)
        final_output = self.projection_layer(merged_embeddings)
        return nn.functional.softplus(final_output)
        #return final_output, mixture_weights

if __name__ == "__main__":
    print(MLP_Embedding(16))
    print(MLP_Embedding(32))
    print(MLP_Embedding(33))
    print(MLP_Embedding(31))
    print(MLP_Embedding(15))
    print(MLP_Embedding(17))
    print(MLP_Embedding(6500))
    print(MLP_Embedding(187))
    FEATURE_SPLITS = [
        list(range(0, 80)),    # Features for MLP #1
        list(range(80, 160)),  # Features for MLP #2
        list(range(160, 240))  # Features for MLP #3
    ]
    TOTAL_INPUT_DIM = 240
    OUTPUT_DIM = 49
    # Embedding dimension is fixed inside the MLP_Embedding class
    BATCH_SIZE = 256
    NUM_SAMPLES = 20000
    EPOCHS = 200

    # Instantiate the new model
    model = DLDSC(FEATURE_SPLITS, OUTPUT_DIM, no_mix_weights=True)
    print("DLDSC Model Architecture:")
    print(model)

    # --- NEW: Generate dummy data where classes depend on ALL features in a group ---
    X_train = torch.randn(NUM_SAMPLES, TOTAL_INPUT_DIM)
    y_train = torch.zeros(NUM_SAMPLES, OUTPUT_DIM)

    # Define the number of classes that depend on group 1 vs group 3
    num_classes_group1 = 25
    num_classes_group3 = OUTPUT_DIM - num_classes_group1

    # Create random but fixed projection matrices to combine the features
    # These matrices will define the "true" function the model needs to learn.
    group1_weights = torch.randn(80, num_classes_group1)
    group3_weights = torch.randn(80, num_classes_group3)

    # First 25 classes depend heavily on the first feature group (MLP #1)
    # Create a linear combination of all 80 features in the group, then apply a non-linearity.
    X_group1 = X_train[:, FEATURE_SPLITS[0]]
    y_train[:, :num_classes_group1] = X_group1 @ group1_weights

    # The remaining classes depend heavily on the third feature group (MLP #3)
    X_group2 = X_train[:, FEATURE_SPLITS[2]]
    y_train[:, num_classes_group1:] = X_group2 @ group3_weights

    # Since the model now outputs positive values, we'll make the target positive
    #y_train = F.relu(y_train) * 2 + 0.2 * torch.rand(NUM_SAMPLES, OUTPUT_DIM)


    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            x_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

    print("--- Training Finished ---")
    torch.save(model.state_dict(), "/home/davidwang/tmp/moe_sim2.pth")
    # --- 5. Inspecting the Learned Mixture Proportions ---
    print("\n--- Inspecting Final Learned Weights for Specific Labels ---")
    model.eval()
    final_mixture_weights = model.get_mixture_props()
    final_weights_np = final_mixture_weights.detach().numpy()
    print(final_weights_np[0:25,:])
    print(final_weights_np[25:,:])
    def print_weights_for_label(label_idx):
        weights = final_weights_np[label_idx]
        print(f"\nMixture Proportions for Label #{label_idx}:")
        for i, weight in enumerate(weights):
            feature_indices = model.feature_splits[i]
            print(f"  - MLP #{i+1} (Features {feature_indices[0]}-{feature_indices[-1]}): {weight:.4f}")
        most_important = np.argmax(weights)
        print(f"  -> Most important MLP for this label: MLP #{most_important + 1}")

    # Check a label that should depend on MLP #1
    print_weights_for_label(5)

    # Check a label that should depend on MLP #3
    print_weights_for_label(30)

    # Check another label that should depend on MLP #1
    print_weights_for_label(15)

    # Check another label that should depend on MLP #3
    print_weights_for_label(45)