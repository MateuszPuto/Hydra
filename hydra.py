import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set a random seed for reproducibility
torch.manual_seed(42)

# 1. Define the Hybrid CNN-GNN Model
class HybridModel(torch.nn.Module):
    """
    A hybrid model combining a Convolutional Neural Network (CNN) for feature
    extraction and a Graph Attention Network (GAT) for relational learning.
    """
    def __init__(self, cnn_out_channels, gnn_in_channels, gnn_out_channels, num_classes):
        super().__init__()
        
        # Define the CNN part for local feature extraction
        # A simple CNN block for demonstration
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, cnn_out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        # Define the GNN part for relational learning (Graph Attention Network)
        # Using a GATv2Conv layer which is more expressive than standard GAT
        self.gnn1 = GATv2Conv(gnn_in_channels, 64, heads=2)
        self.gnn2 = GATv2Conv(64 * 2, gnn_out_channels, heads=1)
        
        # Define the final linear layer for classification
        self.classifier = Linear(gnn_out_channels, num_classes)
        
    def forward(self, x, edge_index):
        """
        Defines the forward pass of the hybrid model.
        Args:
            x (Tensor): Input tensor for the CNN (e.g., image batch).
            edge_index (Tensor): Graph connectivity in the form of edge indices for the GNN.
        """
        # CNN forward pass: extract features from the image grid
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the feature maps to create node features for the GNN
        # Each "node" corresponds to a feature from the flattened map.
        num_features = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, num_features) # Reshape to [batch_size, num_features]
        
        # GNN forward pass: process the graph-structured data
        # Note: We are using a simple fully-connected graph for demonstration.
        x = F.relu(self.gnn1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gnn2(x, edge_index)
        
        # Final classification layer
        return self.classifier(x)

# 2. Prepare Data and Helper Functions
def create_graph_from_features(features):
    """
    Creates a fully connected graph from a set of features.
    
    In a real-world scenario, you would use a more intelligent method to
    construct the graph (e.g., based on feature similarity, spatial proximity, etc.).
    For this example, we assume every feature is connected to every other feature.
    
    Args:
        features (Tensor): A tensor of features, where each feature will be a node.
    Returns:
        Data: A PyTorch Geometric Data object with features and a fully connected edge index.
    """
    num_nodes = features.shape[0]
    # Create all possible pairs of edges for a fully connected graph
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    edge_index = torch.stack([row, col], dim=0)
    
    # Create the PyG Data object
    graph_data = Data(x=features, edge_index=edge_index)
    
    return graph_data

def visualize_graph(data, title):
    """
    Visualizes a simple graph using NetworkX and Matplotlib.
    
    Args:
        data (Data): A PyTorch Geometric Data object.
        title (str): Title for the plot.
    """
    try:
        # Check for a single graph instance (not a batch)
        if len(data.batch) > 1:
            print("Cannot visualize a batch of graphs. Skipping visualization.")
            return

        g = to_networkx(data, to_undirected=True)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx(g, with_labels=False, node_color='lightblue', edge_color='gray', node_size=100)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error visualizing graph: {e}")

# 3. Main Training and Evaluation Loop
if __name__ == '__main__':
    # Load and prepare the MNIST dataset
    dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model, optimizer, and loss function
    # CNN output size for MNIST (28x28) after two conv+pool layers:
    # 28 -> 14 (pool) -> 7 (pool)
    # The output feature maps will be 7x7
    cnn_out_channels = 32 # Arbitrary number of output channels from the CNN
    gnn_in_channels = cnn_out_channels * 7 * 7 # The size of the flattened feature vector
    gnn_out_channels = 128
    num_classes = 10
    
    model = HybridModel(cnn_out_channels, gnn_in_channels, gnn_out_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Model architecture:\n{model}")
    print("\nStarting training loop...")

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm to show a progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Create a fully connected graph for the batch
            num_nodes_in_batch = images.shape[0]
            row = torch.arange(num_nodes_in_batch).repeat_interleave(num_nodes_in_batch).to(device)
            col = torch.arange(num_nodes_in_batch).repeat(num_nodes_in_batch).to(device)
            edge_index = torch.stack([row, col], dim=0)

            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            out = model(images, edge_index)
            loss = criterion(out, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_loss:.4f}")

    # Evaluation loop
    print("\nStarting evaluation...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # Create a fully connected graph for the single image
            num_nodes = images.shape[0]
            row = torch.arange(num_nodes).repeat_interleave(num_nodes).to(device)
            col = torch.arange(num_nodes).repeat(num_nodes).to(device)
            edge_index = torch.stack([row, col], dim=0)

            images, labels = images.to(device), labels.to(device)
            outputs = model(images, edge_index)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nAccuracy on the test dataset: {accuracy:.2f}%")
    
    # Optional: Visualize the graph structure for a single image
    print("\nCreating a sample graph visualization (might take a moment)...")
    sample_images, _ = next(iter(train_loader))
    sample_features = model.conv_layers(sample_images[0:1]).view(-1, gnn_in_channels)
    sample_graph = create_graph_from_features(sample_features[0:1])
    visualize_graph(sample_graph, "Sample Feature Graph (Fully Connected)")
