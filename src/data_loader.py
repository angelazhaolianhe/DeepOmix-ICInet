# src/data_loader.py

import pandas as pd
import numpy as np
import os

class ICINetDataLoader:
    """
    ICINetDataLoader class.
    Responsible for loading gene expression data, patient response data,
    and performing initial preprocessing.
    """
    def __init__(self, data_root_dir="data"):
        """
        Initializes the data loader.

        Args:
            data_root_dir (str): The root directory where data files are located.
                                 Example: "ICInet_Replication/data"
        """
        self.data_root_dir = data_root_dir
        self.gene_expression_dir = os.path.join(data_root_dir, "gene_expression")
        self.patient_response_dir = os.path.join(data_root_dir, "patient_response")
        self.networks_dir = os.path.join(data_root_dir, "networks")

        # Ensure data directories exist
        os.makedirs(self.gene_expression_dir, exist_ok=True)
        os.makedirs(self.patient_response_dir, exist_ok=True)
        os.makedirs(self.networks_dir, exist_ok=True)

        print(f"Data loader initialized. Data root directory: {self.data_root_dir}")

    def load_gene_expression_data(self, cohort_name: str) -> pd.DataFrame:
        """
        Loads gene expression data for a specified cohort.

        Expected file format:
        - CSV file named "{cohort_name}_gene_expression.csv"
        - The first column should be 'patient_id' (or similar unique identifier)
          and will be used as the DataFrame index.
        - Subsequent columns should be gene IDs, and their values are the
          normalized gene expression levels.

        Example (Gide_gene_expression.csv):
        patient_id,gene_0001,gene_0002,...,gene_N
        patient_01,1.23,0.45,...,2.10
        patient_02,0.98,1.76,...,0.55
        ...

        Args:
            cohort_name (str): The name of the cohort (e.g., "Gide", "Liu", "Kim").

        Returns:
            pd.DataFrame: Gene expression data, with patient IDs as index and
                          gene IDs as columns. Returns an empty DataFrame if
                          the file does not exist or an error occurs.
        """
        file_path = os.path.join(self.gene_expression_dir, f"{cohort_name}_gene_expression.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Gene expression data file '{file_path}' not found. "
                  f"Please ensure the file exists in this path.")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path, index_col=0)
            print(f"Successfully loaded {cohort_name} gene expression data. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading {cohort_name} gene expression data from '{file_path}': {e}")
            return pd.DataFrame()

    def load_patient_response_data(self, cohort_name: str) -> pd.Series:
        """
        Loads patient response data for a specified cohort.

        Expected file format:
        - CSV file named "{cohort_name}_patient_response.csv"
        - The first column should be 'patient_id' (or similar unique identifier)
          and will be used as the Series index.
        - The second column should be 'response_status' (or similar),
          containing numerical values (e.g., 0 for non-responder, 1 for responder).

        Example (Gide_patient_response.csv):
        patient_id,response_status
        patient_01,1
        patient_02,0
        ...

        Args:
            cohort_name (str): The name of the cohort (e.g., "Gide", "Liu", "Kim").

        Returns:
            pd.Series: Patient response data, with patient IDs as index and
                       response status as values. Returns an empty Series if
                       the file does not exist or an error occurs.
        """
        file_path = os.path.join(self.patient_response_dir, f"{cohort_name}_patient_response.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Patient response data file '{file_path}' not found. "
                  f"Please ensure the file exists in this path.")
            return pd.Series(dtype=int)
        
        try:
            df = pd.read_csv(file_path, index_col=0)
            if df.shape[1] == 1:
                series = df.iloc[:, 0]
            else:
                raise ValueError("Patient response file is expected to have exactly two columns: "
                                 "patient ID and response status.")
            print(f"Successfully loaded {cohort_name} patient response data. Number of samples: {len(series)}")
            return series
        except Exception as e:
            print(f"Error loading {cohort_name} patient response data from '{file_path}': {e}")
            return pd.Series(dtype=int)

    def combine_data(self, gene_expression_df: pd.DataFrame, patient_response_series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Combines gene expression data and patient response data,
        ensuring patient IDs match. Only patients present in both datasets
        will be included in the combined output.

        Args:
            gene_expression_df (pd.DataFrame): Gene expression data.
            patient_response_series (pd.Series): Patient response data.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Combined gene expression data and
                                            patient response data, filtered to
                                            include only common patients.
                                            Returns empty DataFrames/Series if
                                            input is empty or no common patients are found.
        """
        if gene_expression_df.empty or patient_response_series.empty:
            print("Warning: Gene expression or patient response data is empty. Cannot combine.")
            return pd.DataFrame(), pd.Series(dtype=int)

        # Find common patient IDs
        common_patients = list(set(gene_expression_df.index).intersection(set(patient_response_series.index)))
        if not common_patients:
            print("Warning: No common patient IDs found between gene expression and patient response data.")
            return pd.DataFrame(), pd.Series(dtype=int)

        # Filter and sort data for common patients
        combined_expression = gene_expression_df.loc[common_patients].sort_index()
        combined_response = patient_response_series.loc[common_patients].sort_index()

        print(f"Successfully combined data. Number of common patients: {len(common_patients)}")
        print(f"Combined gene expression data shape: {combined_expression.shape}")
        print(f"Combined patient response data shape: {combined_response.shape}")
        return combined_expression, combined_response

    def load_network_data(self, network_name: str, file_format: str = "gml") -> dict:
        """
        Loads network data (e.g., KEGG, String, PPI).
        This method is a placeholder and assumes a simple file format.
        Actual implementation may require specific parsing based on network type
        (e.g., using networkx for GML, GraphML, or custom parsers for edge lists).

        Expected file formats:
        - GML, GraphML, or CSV (edge list) files.
        - The file name should be "{network_name}_network.{file_format}".

        Example (KEGG_network.gml):
        graph [
          node [ id 0 label "geneA" ]
          node [ id 1 label "geneB" ]
          edge [ source 0 target 1 ]
        ]

        Example (PPI_network.csv - edge list):
        source_gene,target_gene,weight
        geneA,geneB,0.8
        geneC,geneD,0.5
        ...

        Args:
            network_name (str): The name of the network (e.g., "KEGG", "String", "PPI").
            file_format (str): The expected file extension (e.g., "gml", "graphml", "csv").

        Returns:
            dict: Loaded network data (format depends on network type).
                  Returns an empty dictionary if the file does not exist or an error occurs.
        """
        file_path = os.path.join(self.networks_dir, f"{network_name}_network.{file_format}")
        if not os.path.exists(file_path):
            print(f"Warning: Network data file '{file_path}' not found. "
                  f"Please ensure the file exists in this path.")
            return {}
        
        try:
            # Placeholder: In a real scenario, you'd use a library like networkx
            # import networkx as nx
            # if file_format == "gml":
            #     graph = nx.read_gml(file_path)
            # elif file_format == "graphml":
            #     graph = nx.read_graphml(file_path)
            # elif file_format == "csv":
            #     graph = pd.read_csv(file_path) # Assuming edge list
            # else:
            #     raise ValueError(f"Unsupported network file format: {file_format}")
            # print(f"Successfully loaded {network_name} network data (placeholder loading).")
            # return graph # Return the actual graph object

            print(f"Successfully loaded {network_name} network data (placeholder loading). Path: {file_path}")
            return {"status": "loaded", "path": file_path, "format": file_format}
        except Exception as e:
            print(f"Error loading {network_name} network data from '{file_path}': {e}")
            return {}
