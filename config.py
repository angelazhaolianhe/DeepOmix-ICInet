# config.py

import os

class Config:
    """
    Centralized configuration class for the ICInet project.
    Defines paths, model hyperparameters, training parameters, and more.
    """

    # --- Project Directories ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This assumes config.py is in the root of the project
    # Adjust BASE_DIR if config.py is in 'src' directory (e.path.join(os.path.dirname(BASE_DIR), '..'))
    # For now, let's assume config.py is at the root `ICInet_Project/config.py`

    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_PATIENT_GRAPHS_DIR = os.path.join(DATA_DIR, 'processed_patient_graphs')

    NETWORKS_DIR = os.path.join(RAW_DATA_DIR, 'networks')
    GENE_EXPRESSION_DIR = os.path.join(RAW_DATA_DIR, 'gene_expression')
    PATIENT_RESPONSE_DIR = os.path.join(RAW_DATA_DIR, 'patient_response')
    ICI_TARGET_GENES_FILE = os.path.join(RAW_DATA_DIR, 'ici_target_genes.txt')

    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

    # --- Data Files ---
    # Templates for cohort-specific files
    GENE_EXPRESSION_FILE_TEMPLATE = '{cohort_name}_expression.tsv'
    PATIENT_RESPONSE_FILE_TEMPLATE = '{cohort_name}_response.tsv'

    # Network files (used by graph_builder)
    INTEGRATED_GENE_PATHWAY_NETWORK_NAME = "IntegratedGenePathway"
    INTEGRATED_GENE_PATHWAY_NETWORK_FORMAT = "csv" # e.g., 'csv', 'gml', 'graphml'
    PPI_NETWORK_NAME = "Nichenet_PPI"
    PPI_NETWORK_FORMAT = "gml" # e.g., 'gml', 'graphml', 'csv' (for edge list)

    # --- Graph Building Parameters (used by graph_builder.py) ---
    PAGERANK_TOP_N_GENES = 3000
    PAGERANK_ALPHA = 0.85
    PAGERANK_MAX_ITER = 100
    PAGERANK_TOL = 1e-06
    JACCARD_THRESHOLD = 0.0 # Threshold for Jaccard index to add edges in gene-gene network

    # --- Model Hyperparameters (used by model.py) ---
    NUM_GENES_IN_SUBGRAPH = PAGERANK_TOP_N_GENES # This will be dynamic based on the actual subnetwork size
                                                  # but serves as a placeholder for model initialization
    HIDDEN_DIM_GNN = 128
    NUM_GNN_LAYERS = 2
    HIDDEN_DIM_MLP = 64
    OUTPUT_DIM = 1 # Binary classification (responder/non-responder)
    DROPOUT_RATE = 0.5

    # --- Training Parameters (used by main.py) ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5 # L2 regularization
    METRICS_THRESHOLD = 0.5 # Threshold for binary classification metrics (F1, Precision, Recall)

    # --- Experiment Settings ---
    RANDOM_SEED = 42
    # Specify the cohorts you want to include in your analysis
    # Ensure these match the actual folder/file names in raw_data_dir
    ALL_COHORTS = ['Gide', 'Kim', 'Liu', 'Aus', 'Prat', 'Riaz', 'Huang'] # From paper
    TRAIN_COHORTS = ['Gide', 'Kim', 'Liu'] # Example: training on these
    TEST_COHORTS = ['Aus', 'Prat', 'Riaz', 'Huang'] # Example: testing on these (cross-cohort prediction)

    # Specific ICI target genes mentioned in the paper
    ICI_TARGET_GENES = ["PD1", "PD-L1", "CTLA4"]

    # --- Hardware ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Cross-validation settings ---
    K_FOLD_SPLITS = 5
    # For train/test splits within a cohort (e.g., 80/20)
    TRAIN_TEST_SPLIT_RATIO = 0.8

    def __init__(self):
        # Create all necessary directories if they don't exist
        for dir_path in [self.RAW_DATA_DIR, self.PROCESSED_PATIENT_GRAPHS_DIR,
                         self.NETWORKS_DIR, self.GENE_EXPRESSION_DIR, self.PATIENT_RESPONSE_DIR,
                         self.RESULTS_DIR, self.LOGS_DIR, self.CHECKPOINTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)

        # Update BASE_DIR if config.py is in 'src' folder
        # This correction is important for paths to be correct from project root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_file_dir) == 'src':
            self.BASE_DIR = os.path.join(current_file_dir, '..')
            self._update_paths_relative_to_base_dir()

    def _update_paths_relative_to_base_dir(self):
        """Helper to update all paths based on corrected BASE_DIR."""
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.PROCESSED_PATIENT_GRAPHS_DIR = os.path.join(self.DATA_DIR, 'processed_patient_graphs')

        self.NETWORKS_DIR = os.path.join(self.RAW_DATA_DIR, 'networks')
        self.GENE_EXPRESSION_DIR = os.path.join(self.RAW_DATA_DIR, 'gene_expression')
        self.PATIENT_RESPONSE_DIR = os.path.join(self.RAW_DATA_DIR, 'patient_response')
        self.ICI_TARGET_GENES_FILE = os.path.join(self.RAW_DATA_DIR, 'ici_target_genes.txt')

        self.RESULTS_DIR = os.path.join(self.BASE_DIR, 'results')
        self.LOGS_DIR = os.path.join(self.BASE_DIR, 'logs')
        self.CHECKPOINTS_DIR = os.path.join(self.RESULTS_DIR, 'checkpoints')

        # Re-create directories after path update
        for dir_path in [self.RAW_DATA_DIR, self.PROCESSED_PATIENT_GRAPHS_DIR,
                         self.NETWORKS_DIR, self.GENE_EXPRESSION_DIR, self.PATIENT_RESPONSE_DIR,
                         self.RESULTS_DIR, self.LOGS_DIR, self.CHECKPOINTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)


# Instantiate the Config class
# This allows importing `cfg` from other modules to access configurations
cfg = Config()

# Example of how you might use this in another file:
# from config import cfg
# print(cfg.BATCH_SIZE)
