# src/graph_builder.py

import pandas as pd
import numpy as np
import networkx as nx
import os
from typing import List, Dict, Tuple, Set

# Assuming data_loader is in the same 'src' directory
from .data_loader import ICINetDataLoader 

class GraphBuilder:
    """
    GraphBuilder class.
    Responsible for constructing biological knowledge graphs, identifying
    ICI target-proximal genes, and building functional subnetworks for GNN input.
    """
    def __init__(self, data_loader: ICINetDataLoader, config: Dict = None):
        """
        Initializes the GraphBuilder.

        Args:
            data_loader (ICINetDataLoader): An instance of the ICINetDataLoader.
            config (Dict, optional): Configuration dictionary for graph building parameters.
                                     Defaults to None, using default parameters.
        """
        self.data_loader = data_loader
        self.config = config if config is not None else self._get_default_config()
        print("GraphBuilder initialized.")

    def _get_default_config(self) -> Dict:
        """
        Returns default configuration parameters for graph building.
        """
        return {
            "pagerank_top_n_genes": 3000,  # Hyperparameter from the paper
            "pagerank_alpha": 0.85,        # Default PageRank damping factor
            "pagerank_max_iter": 100,      # Default PageRank max iterations
            "pagerank_tol": 1e-06,         # Default PageRank tolerance
            "integrated_gene_pathway_network_name": "IntegratedGenePathway",
            "integrated_gene_pathway_network_format": "csv", # Example: csv for edge list
            "ppi_network_name": "PPI",
            "ppi_network_format": "gml",   # Example: gml for PPI network
            "gene_ontology_network_name": "GeneOntology",
            "gene_ontology_network_format": "gml", # Example: gml for GO network
            "kegg_network_name": "KEGG",
            "kegg_network_format": "gml",
            "string_network_name": "String",
            "string_network_format": "gml",
            "jaccard_threshold": 0.0 # Threshold for Jaccard index to add edges in gene-gene network
        }

    def load_and_process_integrated_gene_pathway_network(self) -> nx.Graph:
        """
        Loads and processes the Integrated Gene Graph (IGG) which is described
        as a bipartite graph where an edge links a gene to a pathway term.
        This is a crucial step for the "prior knowledge" injection.

        USER ACTION REQUIRED:
        - You need to provide the actual integrated gene-pathway network file
          in self.data_loader.networks_dir.
        - This network should combine Gene Ontology and the "13 different prior
          knowledge information databases" mentioned in the paper.
        - The format should match self.config["integrated_gene_pathway_network_format"].

        Expected file format (example for CSV edge list):
        - CSV file named "IntegratedGenePathway_network.csv" (or specified name/format)
        - Columns: 'gene_id', 'pathway_id' (representing gene-pathway links)

        Example (IntegratedGenePathway_network.csv):
        gene_id,pathway_id
        geneA,pathway1
        geneA,pathway2
        geneB,pathway1
        geneC,pathway3
        ...

        Returns:
            nx.Graph: The loaded and processed bipartite gene-pathway graph.
                      Returns an empty graph if the file is not found or an error occurs.
        """
        network_name = self.config["integrated_gene_pathway_network_name"]
        file_format = self.config["integrated_gene_pathway_network_format"]
        file_path = os.path.join(self.data_loader.networks_dir, f"{network_name}_network.{file_format}")

        if not os.path.exists(file_path):
            print(f"Warning: Integrated Gene-Pathway network file '{file_path}' not found. "
                  f"Please provide this file as specified in the paper.")
            return nx.Graph()

        try:
            if file_format == "csv":
                df = pd.read_csv(file_path)
                if 'gene_id' not in df.columns or 'pathway_id' not in df.columns:
                    raise ValueError("CSV for Integrated Gene-Pathway network must have 'gene_id' and 'pathway_id' columns.")
                
                G = nx.Graph()
                # Add nodes with 'type' attribute to distinguish genes and pathways
                for _, row in df.iterrows():
                    gene = row['gene_id']
                    pathway = row['pathway_id']
                    G.add_node(gene, type='gene')
                    G.add_node(pathway, type='pathway')
                    G.add_edge(gene, pathway)
                print(f"Successfully loaded and processed {network_name} network from CSV. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
                return G
            else:
                print(f"Error: Unsupported format '{file_format}' for Integrated Gene-Pathway network. "
                      f"Please implement specific loading logic for this format.")
                return nx.Graph()
        except Exception as e:
            print(f"Error loading/processing {network_name} network from '{file_path}': {e}")
            return nx.Graph()

    def build_gene_gene_network_from_gene_pathway(self, gene_pathway_graph: nx.Graph) -> nx.Graph:
        """
        Builds a gene-gene network from the bipartite gene-pathway graph.
        Edges between genes are weighted by the Jaccard index of their shared pathways.
        This represents the "Integrated Gene Graph (IGG)" used for gene perturbation embeddings.

        Args:
            gene_pathway_graph (nx.Graph): The bipartite graph linking genes to pathways.

        Returns:
            nx.Graph: A gene-gene network with Jaccard similarity as edge weights.
        """
        if gene_pathway_graph.is_empty():
            print("Warning: Input gene-pathway graph is empty. Cannot build gene-gene network.")
            return nx.Graph()

        genes = [n for n, data in gene_pathway_graph.nodes(data=True) if data.get('type') == 'gene']
        pathways_by_gene = {}
        for gene in genes:
            pathways_by_gene[gene] = set(gene_pathway_graph.neighbors(gene))

        gene_gene_G = nx.Graph()
        gene_gene_G.add_nodes_from(genes)

        print(f"Building gene-gene network from {len(genes)} genes...")
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i+1:]:
                set1 = pathways_by_gene.get(gene1, set())
                set2 = pathways_by_gene.get(gene2, set())
                
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                if union > 0:
                    jaccard_index = intersection / union
                    if jaccard_index > self.config["jaccard_threshold"]:
                        gene_gene_G.add_edge(gene1, gene2, weight=jaccard_index)
        
        print(f"Gene-gene network built. Nodes: {gene_gene_G.number_of_nodes()}, Edges: {gene_gene_G.number_of_edges()}")
        return gene_gene_G

    def identify_ici_target_proximal_genes(self, 
                                            gene_network: nx.Graph, 
                                            ici_target_genes: List[str]) -> List[str]:
        """
        Identifies ICI target-proximal genes using the PageRank algorithm.
        The paper uses ICI targets as anchor nodes (personalization parameter = 1)
        and other genes as 0.

        Args:
            gene_network (nx.Graph): The gene-gene network (e.g., PPI or the derived IGG).
            ici_target_genes (List[str]): A list of gene IDs that are ICI targets
                                          (e.g., "PD1", "PD-L1", "CTLA4").

        Returns:
            List[str]: A list of gene IDs identified as ICI target-proximal genes,
                       sorted by PageRank score in descending order, up to top_n.
        """
        if gene_network.is_empty():
            print("Warning: Input gene network is empty. Cannot identify ICI target-proximal genes.")
            return []
        if not ici_target_genes:
            print("Warning: No ICI target genes provided. Cannot identify proximal genes.")
            return []

        # Ensure all target genes are in the network
        valid_ici_targets = [g for g in ici_target_genes if g in gene_network]
        if not valid_ici_targets:
            print("Warning: None of the provided ICI target genes are found in the network.")
            return []

        # Set personalization vector for PageRank
        personalization = {node: 0 for node in gene_network.nodes()}
        for target_gene in valid_ici_targets:
            personalization[target_gene] = 1.0 # Anchor nodes

        print(f"Running PageRank to identify proximal genes for {valid_ici_targets}...")
        try:
            pagerank_scores = nx.pagerank(
                gene_network,
                alpha=self.config["pagerank_alpha"],
                personalization=personalization,
                max_iter=self.config["pagerank_max_iter"],
                tol=self.config["pagerank_tol"],
                weight='weight' # Use edge weights if available
            )
        except nx.NetworkXError as e:
            print(f"Error running PageRank: {e}. This might happen if the graph is disconnected.")
            # Fallback: if PageRank fails, return all nodes or handle differently
            return list(gene_network.nodes())

        # Sort genes by PageRank score and select top N
        sorted_genes = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
        top_n_genes = [gene for gene, score in sorted_genes[:self.config["pagerank_top_n_genes"]]]

        print(f"Identified {len(top_n_genes)} ICI target-proximal genes (top {self.config['pagerank_top_n_genes']}).")
        return top_n_genes

    def extract_subnetwork(self, full_network: nx.Graph, gene_list: List[str]) -> nx.Graph:
        """
        Extracts a subnetwork from a larger graph based on a list of genes.

        Args:
            full_network (nx.Graph): The complete gene network.
            gene_list (List[str]): A list of gene IDs to include in the subnetwork.

        Returns:
            nx.Graph: The induced subnetwork containing only the specified genes
                      and the edges between them.
        """
        if full_network.is_empty() or not gene_list:
            print("Warning: Full network is empty or gene list is empty. Cannot extract subnetwork.")
            return nx.Graph()

        # Filter gene_list to only include nodes present in the full_network
        valid_genes = [gene for gene in gene_list if gene in full_network]
        if not valid_genes:
            print("Warning: None of the genes in the provided list are found in the full network.")
            return nx.Graph()

        subgraph = full_network.subgraph(valid_genes).copy() # .copy() to ensure it's a new graph
        print(f"Extracted subnetwork with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
        return subgraph

    def build_ici_subnetwork_for_gnn(self, 
                                      ici_target_genes: List[str],
                                      use_ppi_for_pagerank: bool = True) -> nx.Graph:
        """
        Orchestrates the process of building the ICI-response-associated subnetwork
        that will be used as input for the Graph Neural Network.

        Steps:
        1. Load the Integrated Gene-Pathway Network (IGG).
        2. Build the Gene-Gene Network from the IGG (based on Jaccard similarity).
        3. (Optional) Load a Protein-Protein Interaction (PPI) network.
        4. Identify ICI target-proximal genes using PageRank on either the
           derived gene-gene network or the PPI network.
        5. Extract the subnetwork using these proximal genes from the derived gene-gene network.

        Args:
            ici_target_genes (List[str]): List of ICI target gene symbols (e.g., ["PD1", "PD-L1"]).
            use_ppi_for_pagerank (bool): If True, use the PPI network for PageRank.
                                         Otherwise, use the gene-gene network derived from IGG.

        Returns:
            nx.Graph: The final ICI-response-associated subnetwork for GNN input.
                      Returns an empty graph if any step fails.
        """
        print("\n--- Building ICI-response-associated subnetwork for GNN ---")

        # Step 1: Load the Integrated Gene-Pathway Network (IGG)
        gene_pathway_igg = self.load_and_process_integrated_gene_pathway_network()
        if gene_pathway_igg.is_empty():
            print("Failed to load Integrated Gene-Pathway Network. Aborting subnetwork build.")
            return nx.Graph()

        # Step 2: Build the Gene-Gene Network from the IGG
        gene_gene_igg = self.build_gene_gene_network_from_gene_pathway(gene_pathway_igg)
        if gene_gene_igg.is_empty():
            print("Failed to build Gene-Gene Network from IGG. Aborting subnetwork build.")
            return nx.Graph()

        # Determine which network to use for PageRank
        pagerank_network = gene_gene_igg
        if use_ppi_for_pagerank:
            print("Attempting to load PPI network for PageRank...")
            ppi_network_name = self.config["ppi_network_name"]
            ppi_network_format = self.config["ppi_network_format"]
            ppi_network_path = os.path.join(self.data_loader.networks_dir, f"{ppi_network_name}_network.{ppi_network_format}")

            if not os.path.exists(ppi_network_path):
                print(f"Warning: PPI network file '{ppi_network_path}' not found. "
                      f"Falling back to gene-gene network derived from IGG for PageRank.")
            else:
                try:
                    # Placeholder for actual PPI network loading (e.g., using networkx.read_gml)
                    # For demonstration, let's assume it's a simple graph
                    if ppi_network_format == "gml":
                        loaded_ppi = nx.read_gml(ppi_network_path)
                    elif ppi_network_format == "graphml":
                        loaded_ppi = nx.read_graphml(ppi_network_path)
                    elif ppi_network_format == "csv": # Assuming edge list: gene1,gene2,weight
                        ppi_df = pd.read_csv(ppi_network_path)
                        loaded_ppi = nx.from_pandas_edgelist(ppi_df, source=ppi_df.columns[0], target=ppi_df.columns[1], edge_attr=True)
                    else:
                        raise ValueError(f"Unsupported PPI network format: {ppi_network_format}")
                    
                    pagerank_network = loaded_ppi
                    print(f"Successfully loaded PPI network. Nodes: {pagerank_network.number_of_nodes()}, Edges: {pagerank_network.number_of_edges()}")
                except Exception as e:
                    print(f"Error loading PPI network: {e}. Falling back to gene-gene network derived from IGG for PageRank.")
                    pagerank_network = gene_gene_igg


        # Step 3: Identify ICI target-proximal genes
        proximal_genes = self.identify_ici_target_proximal_genes(pagerank_network, ici_target_genes)
        if not proximal_genes:
            print("No ICI target-proximal genes identified. Aborting subnetwork build.")
            return nx.Graph()

        # Step 4: Extract the subnetwork using these proximal genes from the gene-gene IGG
        # The paper states: "we use the selected functional subnetworks as prior knowledge networks
        # for graph neural network training input for immune therapy response prediction modeling."
        # This implies the final GNN input graph is derived from the prior knowledge.
        final_gnn_subnetwork = self.extract_subnetwork(gene_gene_igg, proximal_genes)
        
        if final_gnn_subnetwork.is_empty():
            print("Final GNN subnetwork is empty. Aborting.")
            return nx.Graph()

        print("ICI-response-associated subnetwork for GNN built successfully.")
        return final_gnn_subnetwork

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Setup dummy data for testing GraphBuilder
    # Create dummy data directories if they don't exist
    os.makedirs("data/gene_expression", exist_ok=True)
    os.makedirs("data/patient_response", exist_ok=True)
    os.makedirs("data/networks", exist_ok=True)

    # Create a dummy Integrated Gene-Pathway Network CSV
    # gene_id,pathway_id
    # geneA,path1
    # geneA,path2
    # geneB,path1
    # geneC,path3
    # geneD,path4
    # geneE,path1
    # geneE,path5
    # geneF,path6
    dummy_igg_data = {
        'gene_id': ['geneA', 'geneA', 'geneB', 'geneC', 'geneD', 'geneE', 'geneE', 'geneF'],
        'pathway_id': ['path1', 'path2', 'path1', 'path3', 'path4', 'path1', 'path5', 'path6']
    }
    pd.DataFrame(dummy_igg_data).to_csv("data/networks/IntegratedGenePathway_network.csv", index=False)

    # Create a dummy PPI network (GML format for example)
    # This is a simple example, real GML files are more complex
    dummy_ppi_gml_content = """graph [
  node [ id "geneA" label "geneA" ]
  node [ id "geneB" label "geneB" ]
  node [ id "geneC" label "geneC" ]
  node [ id "geneD" label "geneD" ]
  node [ id "geneE" label "geneE" ]
  node [ id "geneF" label "geneF" ]
  edge [ source "geneA" target "geneB" weight 0.9 ]
  edge [ source "geneA" target "geneE" weight 0.7 ]
  edge [ source "geneB" target "geneC" weight 0.6 ]
  edge [ source "geneD" target "geneF" weight 0.5 ]
]"""
    with open("data/networks/PPI_network.gml", "w") as f:
        f.write(dummy_ppi_gml_content)

    print("\n--- Testing GraphBuilder ---")
    data_loader = ICINetDataLoader() # Initialize data loader
    graph_builder = GraphBuilder(data_loader)

    ici_targets = ["geneA", "geneC"] # Example ICI target genes

    # Build subnetwork using PPI for PageRank
    print("\n--- Building subnetwork (using PPI for PageRank) ---")
    gnn_subnetwork_ppi = graph_builder.build_ici_subnetwork_for_gnn(ici_targets, use_ppi_for_pagerank=True)
    print(f"GNN Subnetwork (PPI-based PageRank) nodes: {gnn_subnetwork_ppi.number_of_nodes()}, edges: {gnn_subnetwork_ppi.number_of_edges()}")
    if not gnn_subnetwork_ppi.is_empty():
        print("Nodes in GNN subnetwork (PPI-based PageRank):", list(gnn_subnetwork_ppi.nodes()))

    # Build subnetwork using IGG-derived gene-gene network for PageRank
    print("\n--- Building subnetwork (using IGG-derived gene-gene network for PageRank) ---")
    gnn_subnetwork_igg = graph_builder.build_ici_subnetwork_for_gnn(ici_targets, use_ppi_for_pagerank=False)
    print(f"GNN Subnetwork (IGG-based PageRank) nodes: {gnn_subnetwork_igg.number_of_nodes()}, edges: {gnn_subnetwork_igg.number_of_edges()}")
    if not gnn_subnetwork_igg.is_empty():
        print("Nodes in GNN subnetwork (IGG-based PageRank):", list(gnn_subnetwork_igg.nodes()))

    # Clean up dummy data files and directories
    os.remove("data/networks/IntegratedGenePathway_network.csv")
    os.remove("data/networks/PPI_network.gml")
    os.rmdir("data/gene_expression") # These might be empty, rmdir will fail if not empty
    os.rmdir("data/patient_response")
    os.rmdir("data/networks")
    os.rmdir("data")
    print("\n--- Dummy data files and directories cleaned up ---")
