import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from typing import List, Dict, Tuple, Set

# Assuming data_loader is in the same 'src' directory
from .data_loader import ICINetDataLoader # To get data directories
# Assuming graph_builder is in the same 'src' directory
from .graph_builder import GraphBuilder # To potentially use its config or methods, though we'll pass the graph


class PatientDataPreprocessingConfig:
    """
    Configuration parameters for patient-specific data preprocessing.
    """
    # --- Input Data Paths ---
    # These paths will be used by the ICINetDataLoader, which this preprocessor relies on.
    # We will get data paths from data_loader instance passed in __init__.
    GENE_EXPRESSION_FILE_TEMPLATE = '{cohort_name}_expression.tsv'
    PATIENT_RESPONSE_FILE_TEMPLATE = '{cohort_name}_response.tsv'

    # --- Output Data Paths ---
    PROCESSED_DATA_DIR = 'data/processed_patient_graphs' # Where to save the final PyG Data objects
    # Will save processed PyG Data objects for each patient in subdirectories
    # e.g., data/processed_patient_graphs/Gide/patient_1.pt

    # --- Preprocessing Parameters ---
    # No specific filtering here, assumes raw data is clean or filtered by data_loader
    # Gene filtering (e.g., variance) should ideally happen before or in data_loader,
    # or by restricting to genes present in the ICI subnetwork.
    MIN_GENE_EXPRESSION_VALUE = 0.0 # Example: Cap lower values, if needed


class PatientDataProcessor:
    """
    PatientDataProcessor class.
    Responsible for loading patient-specific gene expression and response data,
    and transforming it into PyTorch Geometric Data objects for each patient,
    aligned with a pre-built knowledge graph.
    """
    def __init__(self, data_loader: ICINetDataLoader, config: Dict = None):
        """
        Initializes the PatientDataProcessor.

        Args:
            data_loader (ICINetDataLoader): An instance of the ICINetDataLoader.
            config (Dict, optional): Configuration dictionary for preprocessing parameters.
                                     Defaults to None, using default parameters.
        """
        self.data_loader = data_loader
        self.config = config if config is not None else PatientDataPreprocessingConfig.__dict__
        # Create output directory if it doesn't exist
        os.makedirs(self.config['PROCESSED_DATA_DIR'], exist_ok=True)
        print("PatientDataProcessor initialized.")

    def _load_gene_expression_for_cohort(self, cohort_name: str) -> pd.DataFrame:
        """
        Loads gene expression data for a specific cohort from the data_loader.
        Assumes data_loader handles finding the correct file path.
        Expected format: Genes as index, Patients as columns. Values are expression levels.
        """
        filepath = os.path.join(self.data_loader.gene_expression_dir,
                                self.config['GENE_EXPRESSION_FILE_TEMPLATE'].format(cohort_name=cohort_name))
        print(f"Loading gene expression for {cohort_name} from {filepath}...")
        try:
            df_expr = pd.read_csv(filepath, sep='\t', index_col=0)
            df_expr.index = df_expr.index.astype(str) # Ensure gene names are strings
            print(f"Loaded {df_expr.shape[0]} genes, {df_expr.shape[1]} patients.")
            return df_expr
        except FileNotFoundError:
            print(f"Error: Gene expression file not found at {filepath}")
            return pd.DataFrame() # Return empty DataFrame on error

    def _load_patient_response_for_cohort(self, cohort_name: str) -> pd.DataFrame:
        """
        Loads patient response data for a specific cohort from the data_loader.
        Assumes data_loader handles finding the correct file path.
        Expected format: PatientID, ResponseStatus (e.g., 'Responder', 'Non-responder')
        """
        filepath = os.path.join(self.data_loader.patient_response_dir,
                                self.config['PATIENT_RESPONSE_FILE_TEMPLATE'].format(cohort_name=cohort_name))
        print(f"Loading patient response for {cohort_name} from {filepath}...")
        try:
            df_response = pd.read_csv(filepath, sep='\t', index_col=0)
            df_response.index = df_response.index.astype(str) # Patient IDs as strings
            # Standardize response labels to 0/1
            # Assuming 'Responder' (or similar like 'CR', 'PR') is 1, others 0
            df_response['label'] = df_response['ResponseStatus'].apply(
                lambda x: 1 if x in ['Responder', 'CR', 'PR'] else 0
            )
            print(f"Loaded {df_response.shape[0]} patient responses.")
            return df_response[['label']]
        except FileNotFoundError:
            print(f"Error: Patient response file not found at {filepath}")
            return pd.DataFrame() # Return empty DataFrame on error

    def process_cohort_data(self,
                            cohort_name: str,
                            ici_subnetwork_graph: nx.Graph,
                            ici_proximal_genes: List[str]) -> List[Data]:
        """
        Main method to process a single cancer cohort's data into PyTorch Geometric
        Data objects, using the provided ICI subnetwork.

        Args:
            cohort_name (str): The name of the cancer cohort (e.g., 'Gide', 'Kim').
            ici_subnetwork_graph (nx.Graph): The pre-built ICI-response-associated subnetwork
                                              (e.g., from GraphBuilder.build_ici_subnetwork_for_gnn).
            ici_proximal_genes (List[str]): The list of genes that constitute the nodes
                                             of the ici_subnetwork_graph, in the desired order.

        Returns:
            List[Data]: A list of PyTorch Geometric Data objects, one for each patient.
                        Returns an empty list if processing fails or no common data.
        """
        print(f"\n--- Processing patient data for cohort: {cohort_name} ---")

        if ici_subnetwork_graph.is_empty() or not ici_proximal_genes:
            print("Error: ICI subnetwork graph or proximal genes list is empty. Cannot process cohort data.")
            return []

        # Load patient-specific data
        df_expr = self._load_gene_expression_for_cohort(cohort_name)
        df_response = self._load_patient_response_for_cohort(cohort_name)

        if df_expr.empty or df_response.empty:
            print(f"Skipping preprocessing for cohort {cohort_name} due to missing or empty data.")
            return []

        # Align patient IDs
        common_patients = list(set(df_expr.columns).intersection(df_response.index))
        if not common_patients:
            print(f"No common patients found between expression and response data for {cohort_name}. Skipping.")
            return []

        df_expr_filtered = df_expr[common_patients]
        df_response_filtered = df_response.loc[common_patients]
        print(f"Found {len(common_patients)} patients with complete data for {cohort_name}.")

        # --- Prepare the graph components (x, edge_index, y) ---
        # 1. edge_index: Convert the NetworkX graph to PyTorch Geometric format
        #    Ensure node order matches the `ici_proximal_genes` list for feature alignment (x)
        subnetwork_node_mapping = {gene: i for i, gene in enumerate(ici_proximal_genes)}
        
        # Filter edges to only include those between the proximal genes that are actually in the graph
        edge_list_numeric = []
        for u, v in ici_subnetwork_graph.edges():
            if u in subnetwork_node_mapping and v in subnetwork_node_mapping:
                edge_list_numeric.append([subnetwork_node_mapping[u], subnetwork_node_mapping[v]])
        
        if not edge_list_numeric:
            print("Warning: No edges found in the ICI subnetwork after mapping to proximal genes. Graph will be empty.")
            # For isolated nodes, PyG Data can still be created, but GNNs might struggle.
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list_numeric, dtype=torch.long).t().contiguous()
            # Add reverse edges for undirected graph (GNNs often expect this)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = torch.unique(edge_index, dim=1) # Remove duplicates

        # List to store all processed Data objects for the cohort
        processed_patient_data = []
        cohort_output_dir = os.path.join(self.config['PROCESSED_DATA_DIR'], cohort_name)
        os.makedirs(cohort_output_dir, exist_ok=True)

        # 2. x (node features): Gene expression for each patient
        # 3. y (label): Patient response
        for patient_id in tqdm(common_patients, desc=f"Creating PyG Data for {cohort_name}"):
            # Extract gene expression for the current patient,
            # ensuring genes are in the order of ici_proximal_genes
            # Fill NaN if some proximal genes are missing for a patient (e.g., with 0 or mean)
            patient_expr_values = df_expr_filtered[patient_id].reindex(ici_proximal_genes, fill_value=0.0).values
            
            # Apply any final normalization/clipping if specified in config
            # patient_expr_values[patient_expr_values < self.config['MIN_GENE_EXPRESSION_VALUE']] = self.config['MIN_GENE_EXPRESSION_VALUE']

            x = torch.tensor(patient_expr_values, dtype=torch.float).unsqueeze(1) # (num_nodes, 1) features

            # Get patient label
            y = torch.tensor([df_response_filtered.loc[patient_id, 'label']], dtype=torch.float) # (1,) label

            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data.patient_id = patient_id # Store original patient ID for tracking
            data.cohort_name = cohort_name # Store cohort name

            processed_patient_data.append(data)

            # Optionally, save each Data object individually
            torch.save(data, os.path.join(cohort_output_dir, f'{patient_id}.pt'))

        print(f"Finished processing {len(processed_patient_data)} patients for {cohort_name}.")
        return processed_patient_data


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Setting up dummy data for preprocess.py testing ---")
    # Create necessary dummy directories for data_loader
    base_data_dir = "temp_data_for_preprocess_test"
    gene_expr_dir = os.path.join(base_data_dir, "gene_expression")
    patient_resp_dir = os.path.join(base_data_dir, "patient_response")
    networks_dir = os.path.join(base_data_dir, "networks")
    processed_graphs_dir = os.path.join(base_data_dir, "processed_patient_graphs")

    os.makedirs(gene_expr_dir, exist_ok=True)
    os.makedirs(patient_resp_dir, exist_ok=True)
    os.makedirs(networks_dir, exist_ok=True)
    os.makedirs(processed_graphs_dir, exist_ok=True)

    # Dummy ICINetDataLoader (just for paths)
    class DummyICINetDataLoader:
        def __init__(self, base_dir):
            self.base_data_dir = base_dir
            self.gene_expression_dir = os.path.join(base_dir, "gene_expression")
            self.patient_response_dir = os.path.join(base_dir, "patient_response")
            self.networks_dir = os.path.join(base_dir, "networks")

    data_loader = DummyICINetDataLoader(base_data_dir)

    # 1. Create a dummy ICI subnetwork graph (mimicking output from GraphBuilder)
    ici_prox_genes = ["geneA", "geneB", "geneC", "geneD", "geneE"]
    ici_subnetwork = nx.Graph()
    ici_subnetwork.add_edges_from([("geneA", "geneB"), ("geneB", "geneC"), ("geneC", "geneD")])
    ici_subnetwork.add_node("geneE") # An isolated gene in the subnetwork
    print(f"Dummy ICI Subnetwork: {ici_subnetwork.nodes()} nodes, {ici_subnetwork.edges()} edges.")

    # 2. Create dummy gene expression data for a cohort (e.g., 'Gide')
    cohort_name = "Gide"
    dummy_expr_df = pd.DataFrame({
        'patient1': [10.0, 5.0, 15.0, 20.0, 2.0], # geneA, B, C, D, E expression
        'patient2': [12.0, 6.0, 14.0, 18.0, 3.0],
        'patient3': [8.0, 4.0, 16.0, 22.0, 1.5]
    }, index=["geneA", "geneB", "geneC", "geneD", "geneE"])
    dummy_expr_df.to_csv(os.path.join(gene_expr_dir, f"{cohort_name}_expression.tsv"), sep='\t')

    # 3. Create dummy patient response data for the cohort
    dummy_response_df = pd.DataFrame({
        'ResponseStatus': ['Responder', 'Non-responder', 'Responder']
    }, index=['patient1', 'patient2', 'patient3'])
    dummy_response_df.to_csv(os.path.join(patient_resp_dir, f"{cohort_name}_response.tsv"), sep='\t')

    # --- Initialize and run PatientDataProcessor ---
    processor = PatientDataProcessor(data_loader)

    processed_data_list = processor.process_cohort_data(
        cohort_name=cohort_name,
        ici_subnetwork_graph=ici_subnetwork,
        ici_proximal_genes=ici_prox_genes
    )

    print(f"\nTotal processed PyG Data objects: {len(processed_data_list)}")
    if processed_data_list:
        sample_data = processed_data_list[0]
        print(f"Sample PyG Data object (first patient):")
        print(f"  x (gene features) shape: {sample_data.x.shape}")
        print(f"  edge_index shape: {sample_data.edge_index.shape}")
        print(f"  y (label): {sample_data.y.item()}")
        print(f"  Patient ID: {sample_data.patient_id}")
        print(f"  Cohort: {sample_data.cohort_name}")

        # Verify a saved file
        saved_file_path = os.path.join(processed_graphs_dir, cohort_name, f'{processed_data_list[0].patient_id}.pt')
        if os.path.exists(saved_file_path):
            loaded_data = torch.load(saved_file_path)
            print(f"  Successfully loaded saved data from: {saved_file_path}")
            print(f"  Loaded data x shape: {loaded_data.x.shape}")
        else:
            print(f"  Error: Saved file not found at {saved_file_path}")

 
