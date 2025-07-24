import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

class AttentionModule(nn.Module):
    """
    A conceptual attention module for weighting node features.
    The paper mentions an 'attention module, which scores each interval of data'.
    This is a simplified interpretation for demonstration.
    """
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights.data)

    def forward(self, x):
        # x: node features (num_nodes, input_dim)
        # Calculate attention scores for each node feature
        scores = torch.matmul(x, self.attention_weights).squeeze() # (num_nodes,)
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(scores, dim=0) # Normalize across nodes

        # Apply attention: element-wise multiplication of features by their scores
        # Reshape attention_probs to broadcast correctly (num_nodes, 1)
        attended_x = x * attention_probs.unsqueeze(1)
        return attended_x, attention_probs

class GNNEncoder(nn.Module):
    """
    Encodes the gene expression profiles using Graph Convolutional Networks
    leveraging the biological knowledge graph.
    Corresponds to the GNN part that processes 'gi' nodes with 'P' pathways.
    """
    def __init__(self, input_dim, hidden_dim_gnn, num_layers):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()

        # First GCN layer
        self.conv_layers.append(GCNConv(input_dim, hidden_dim_gnn))

        # Subsequent GCN layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim_gnn, hidden_dim_gnn))

        # Simple Attention module
        self.attention = AttentionModule(hidden_dim_gnn)


    def forward(self, x, edge_index):
        # x: input node features (gene expression)
        # edge_index: graph connectivity (from prior knowledge networks)

        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1: # Apply dropout between layers
                x = F.dropout(x, p=0.5, training=self.training)

        # Apply attention after GNN layers (or within, depending on design)
        attended_x, attention_weights = self.attention(x)
        return attended_x, attention_weights

class ICInet(nn.Module):
    """
    The main ICInet model, combining prior knowledge (via GNN)
    with gene expression data to predict immune therapy response.
    """
    def __init__(self, num_genes, hidden_dim_gnn=128, num_gnn_layers=2,
                 hidden_dim_mlp=64, output_dim=1):
        super(ICInet, self).__init__()

        self.num_genes = num_genes

        # 1. GNN Encoder for processing gene expression on the knowledge graph
        # This maps gene expression (num_genes, initial_feature_dim)
        # to latent representations (num_genes, hidden_dim_gnn)
        self.gnn_encoder = GNNEncoder(num_genes, hidden_dim_gnn, num_gnn_layers)

        # 2. Gene Level MLP Layer (as described in Figure 1B)
        # This processes the aggregated or attended GNN output.
        # We'll use a global pooling to get a graph-level representation for the MLP.
        # The input to the MLP will be the output dimension of the GNN encoder.
        self.mlp_head = nn.Sequential(
            Linear(hidden_dim_gnn, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            Linear(hidden_dim_mlp, output_dim)
        )

        # Sigmoid for binary classification probability output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch_map=None):
        """
        Forward pass for the ICInet model.

        Args:
            x (torch.Tensor): Gene expression profiles for all patients/samples.
                              Shape: (total_num_nodes, num_genes)
                              In a batch scenario, total_num_nodes would be sum of nodes across graphs.
                              The paper says 'gene expression profiles' are injected on graph.
                              So, x here represents features for each node (gene) in the graph.
                              A simpler interpretation for a patient-level prediction:
                              x is the gene expression profile for a *single patient*,
                              where x.shape = (num_genes, 1) or (num_genes, feature_dim_per_gene)
                              Let's assume x is (num_genes, 1) or (num_genes, embedding_dim)
                              if genes are embedded. The paper mentions V=(g1...gN) N=10000.
                              So, x is (N, 1) where N is the number of genes and 1 is the expression value.

                              Revised: The paper suggests 'gene expression profiles of cancer patients
                              are injected into gene regulatory networks.' and 'Each node has feature vector
                              formed with 10 000 genes normalized expression vector V = (g1, g2, ..., gN), N = 10000.'
                              This implies that *each node* (which is a gene in the knowledge graph)
                              has a feature vector representing the *expression of that gene across N patients*.
                              This interpretation might be too complex for an invention.

                              A more common interpretation for patient-level prediction using GNN:
                              The graph *nodes* are genes. The *features* on these nodes are the
                              gene expression values for a *given patient*.
                              So, input `x` would be `(num_genes, 1)` representing expression values for one patient.
                              If processing a batch of patients, it would be `(total_nodes_in_batch, 1)`
                              and `batch_map` would indicate which node belongs to which graph.

            edge_index (torch.Tensor): The connectivity of the biological knowledge graph.
                                       Shape: (2, num_edges)
            batch_map (torch.Tensor, optional): Batch vector for `torch_geometric.data.Batch`.
                                                 Maps each node to its graph in the batch.
                                                 Required for `global_mean_pool`. Defaults to None.
        """
        if x.dim() == 1: # If a single patient's gene expression is passed as (num_genes,)
            x = x.unsqueeze(1) # Make it (num_genes, 1)

        # Pass through GNN encoder
        # The GNN processes the gene-level features on the gene-gene graph.
        # Output `gene_embeddings` shape: (num_genes, hidden_dim_gnn)
        gene_embeddings, attention_weights = self.gnn_encoder(x, edge_index)

        # Aggregate gene embeddings to get a patient-level representation.
        # 'batch_map' is needed here if processing multiple patient graphs in a batch.
        # If processing one patient graph at a time, global_mean_pool sums over nodes.
        if batch_map is not None:
            # Aggregate node embeddings per graph in the batch
            # This turns (total_num_nodes, hidden_dim) into (num_graphs_in_batch, hidden_dim)
            patient_representation = global_mean_pool(gene_embeddings, batch_map)
        else:
            # If processing a single patient's graph, mean pool over all genes (nodes)
            patient_representation = torch.mean(gene_embeddings, dim=0, keepdim=True) # (1, hidden_dim_gnn)

        # Pass the patient-level representation through the MLP head
        logits = self.mlp_head(patient_representation)

        # Apply sigmoid for response probability
        response_probability = self.sigmoid(logits)

        return response_probability, attention_weights

if __name__ == '__main__':
    # Define hyper-parameters (these would be tuned)
    num_genes = 10000 
    hidden_dim_gnn = 128
    num_gnn_layers = 2
    hidden_dim_mlp = 64
    output_dim = 1 # For binary classification (responder/non-responder)

    # Instantiate the model
    model = ICInet(num_genes, hidden_dim_gnn, num_gnn_layers, hidden_dim_mlp, output_dim)
    print(model)

    # --- Inventing example data ---
    # In a real scenario, this 'edge_index' would come from your preprocessed
    # biological knowledge graphs (KEGG, String, GO, PPI, etc.).
    # A simple example of a graph with 5 genes (nodes) and 6 edges
    # (gene 0 connects to 1, 1 to 2, etc.)
    num_example_genes = 5
    example_edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 0],  # Source nodes
        [1, 2, 3, 4, 0, 2]   # Target nodes
    ], dtype=torch.long)

    # In the actual ICInet, num_genes would be N=10000 and x would be (10000, 1) for one patient.
    # For this tiny example, let's assume 'num_example_genes' as our N.
    # x: Gene expression profile for one patient. (num_genes, 1)
    # The paper implies 'V=(g1...gN)' as the feature vector for EACH node (gene).
    # So, here, 'x' for the GNN would represent the expression level of each gene.
    # We are simplifying this to be (num_genes, 1) for a single patient.
    # A random expression profile for an example patient
    example_x = torch.randn(num_example_genes, 1) # (num_genes, feature_dim_per_gene)

    # Forward pass (for a single patient)
    # Note: If batching patients, you'd stack x for all patients and create a 'batch_map'.
    # For this example, batch_map=None as it's a single graph.
    response_prob, attention_scores = model(example_x, example_edge_index)

    print("\n--- Example Model Output ---")
    print(f"Input gene expression shape: {example_x.shape}")
    print(f"Graph edge index shape: {example_edge_index.shape}")
    print(f"Predicted immune therapy response probability: {response_prob.item():.4f}")
    print(f"Attention scores shape (per gene): {attention_scores.shape}")
    # You could then analyze `attention_scores` to see which genes were most important.
