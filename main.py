# main.py

import os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
import logging

# Import project modules
from config import cfg # Access global configurations
from src.dataloader import ICINetDataset
from src.graph_builder import GraphBuilder
from src.preprocess import PatientDataProcessor
from src.model import ICInet
from src.utils import set_seed, calculate_metrics, save_checkpoint, setup_logging

def run_preprocessing_pipeline(config):
    """
    Orchestrates the data preprocessing pipeline.
    Builds the ICI subnetwork and processes patient data.
    """
    logging.info("--- Starting Data Preprocessing Pipeline ---")

    # Initialize DataLoader (used by preprocessors to locate raw data)
    # Note: This is a simplified DataLoader that mostly provides paths.
    # The actual PyG DataLoader is used in the training loop.
    data_loader_instance_for_paths = type('DummyDataLoader', (object,), {
        'gene_expression_dir': config.GENE_EXPRESSION_DIR,
        'patient_response_dir': config.PATIENT_RESPONSE_DIR,
        'networks_dir': config.NETWORKS_DIR
    })()

    # 1. Build the knowledge graph and identify ICI-proximal genes
    graph_builder = GraphBuilder(data_loader_instance_for_paths, {
        "pagerank_top_n_genes": config.PAGERANK_TOP_N_GENES,
        "pagerank_alpha": config.PAGERANK_ALPHA,
        "pagerank_max_iter": config.PAGERANK_MAX_ITER,
        "pagerank_tol": config.PAGERANK_TOL,
        "integrated_gene_pathway_network_name": config.INTEGRATED_GENE_PATHWAY_NETWORK_NAME,
        "integrated_gene_pathway_network_format": config.INTEGRATED_GENE_PATHWAY_NETWORK_FORMAT,
        "ppi_network_name": config.PPI_NETWORK_NAME,
        "ppi_network_format": config.PPI_NETWORK_FORMAT,
        "jaccard_threshold": config.JACCARD_THRESHOLD
    })
    
    # The paper's text "We used multiple different gene regulatory networks including KEGG and String"
    # and "filter edges to only include those between the proximal genes that are actually in the graph"
    # suggests the final GNN input graph should be the extracted subnetwork from the IGG.
    # The PageRank could potentially be run on a broader network (like PPI) to identify proximal genes.
    # For now, let's assume `build_ici_subnetwork_for_gnn` handles this complexity.
    
    # Build the final GNN input subnetwork graph based on ICI targets
    # Use PPI network for PageRank, but extract subnetwork from the richer IGG-derived gene-gene network
    ici_subnetwork_graph = graph_builder.build_ici_subnetwork_for_gnn(
        config.ICI_TARGET_GENES, use_ppi_for_pagerank=True
    )
    
    if ici_subnetwork_graph.is_empty():
        logging.error("Failed to build ICI subnetwork. Cannot proceed with preprocessing.")
        return None, None

    # Get the ordered list of genes that form the nodes of the ICI subnetwork
    ici_proximal_genes_ordered = list(ici_subnetwork_graph.nodes())
    
    # Update NUM_GENES_IN_SUBGRAPH in config for model initialization
    config.NUM_GENES_IN_SUBGRAPH = len(ici_proximal_genes_ordered)
    logging.info(f"Final ICI subnetwork has {config.NUM_GENES_IN_SUBGRAPH} genes.")

    # 2. Process patient-specific data into PyG Data objects
    patient_data_processor = PatientDataProcessor(data_loader_instance_for_paths, {
        "PROCESSED_DATA_DIR": config.PROCESSED_PATIENT_GRAPHS_DIR,
        "GENE_EXPRESSION_FILE_TEMPLATE": config.GENE_EXPRESSION_FILE_TEMPLATE,
        "PATIENT_RESPONSE_FILE_TEMPLATE": config.PATIENT_RESPONSE_FILE_TEMPLATE
    })

    # Process each cohort specified in the config
    all_processed_data = []
    for cohort_name in config.ALL_COHORTS:
        processed_cohort_data = patient_data_processor.process_cohort_data(
            cohort_name, ici_subnetwork_graph, ici_proximal_genes_ordered
        )
        all_processed_data.extend(processed_cohort_data)

    logging.info(f"Total processed patient graphs across all cohorts: {len(all_processed_data)}")
    logging.info("--- Data Preprocessing Pipeline Finished ---")
    return ici_subnetwork_graph, ici_proximal_genes_ordered

def train_model(model, train_loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    y_true_all, y_pred_proba_all = [], []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # Ensure x is (num_nodes, 1) and edge_index is (2, num_edges)
        output_proba, _ = model(data.x, data.edge_index, data.batch)
        
        loss = criterion(output_proba.squeeze(), data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_true_all.extend(data.y.cpu().numpy())
        y_pred_proba_all.extend(output_proba.squeeze().cpu().detach().numpy())

    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(np.array(y_true_all), np.array(y_pred_proba_all), cfg.METRICS_THRESHOLD)
    return avg_loss, metrics

def evaluate_model(model, loader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    y_true_all, y_pred_proba_all = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output_proba, _ = model(data.x, data.edge_index, data.batch)
            
            loss = criterion(output_proba.squeeze(), data.y)
            total_loss += loss.item()

            y_true_all.extend(data.y.cpu().numpy())
            y_pred_proba_all.extend(output_proba.squeeze().cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(np.array(y_true_all), np.array(y_pred_proba_all), cfg.METRICS_THRESHOLD)
    return avg_loss, metrics

def run_experiment(config):
    """
    Main function to run the ICInet experiment.
    Handles data loading, model training, and evaluation.
    """
    # Set up logging for the experiment
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOGS_DIR, f"experiment_{current_time}.log")
    setup_logging(log_file)
    logging.info(f"Starting ICInet experiment with config:\n{config.__dict__}")
    set_seed(config.RANDOM_SEED)

    # --- Data Preprocessing ---
    ici_subnetwork_graph, ici_proximal_genes_ordered = run_preprocessing_pipeline(config)
    if ici_subnetwork_graph is None or ici_proximal_genes_ordered is None:
        logging.error("Preprocessing failed. Exiting.")
        return

    # Load all processed data into an ICINetDataset
    full_dataset = ICINetDataset(root=config.PROCESSED_PATIENT_GRAPHS_DIR,
                                 cohort_names=config.ALL_COHORTS)

    # --- Model Initialization ---
    model = ICInet(
        num_genes=config.NUM_GENES_IN_SUBGRAPH, # Number of nodes (genes) in the subgraph
        hidden_dim_gnn=config.HIDDEN_DIM_GNN,
        num_gnn_layers=config.NUM_GNN_LAYERS,
        hidden_dim_mlp=config.HIDDEN_DIM_MLP,
        output_dim=config.OUTPUT_DIM
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for probabilities

    logging.info(f"Model: \n{model}")
    logging.info(f"Optimizer: {optimizer}")
    logging.info(f"Loss Function: {criterion}")
    logging.info(f"Using device: {config.DEVICE}")

    # --- Experiment Scenarios (as described in the paper) ---
    # The paper describes:
    # 1. Intra-cohort prediction (train/test within same cohort)
    # 2. Cross-cohort prediction (train on one/some, test on others)
    # 3. K-fold cross-validation
    # We will implement a general train/test split here and outline others.

    # Example: Cross-cohort prediction
    train_indices = [i for i, data in enumerate(full_dataset) if data.cohort_name in config.TRAIN_COHORTS]
    test_indices = [i for i, data in enumerate(full_dataset) if data.cohort_name in config.TEST_COHORTS]

    if not train_indices or not test_indices:
        logging.warning("No data found for the specified TRAIN_COHORTS or TEST_COHORTS. "
                        "Adjust config.ALL_COHORTS, TRAIN_COHORTS, TEST_COHORTS if this is unexpected.")
        # Fallback to single cohort split if cross-cohort is not possible
        if not train_indices and not test_indices and full_dataset:
            logging.info("Attempting a single cohort train/test split as fallback.")
            train_val_test_split_dataset(full_dataset, model, optimizer, criterion, config)
        return

    train_dataset = full_dataset[train_indices]
    test_dataset = full_dataset[test_indices]

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    logging.info(f"Training on {len(train_dataset)} samples from cohorts: {config.TRAIN_COHORTS}")
    logging.info(f"Testing on {len(test_dataset)} samples from cohorts: {config.TEST_COHORTS}")

    best_val_auc = -1
    best_epoch = 0

    # --- Training Loop ---
    logging.info("\n--- Starting Model Training ---")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_metrics = train_model(model, train_loader, optimizer, criterion, config.DEVICE)
        logging.info(f"Epoch {epoch}/{config.NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Metrics: {train_metrics}")

        val_loss, val_metrics = evaluate_model(model, test_loader, criterion, config.DEVICE) # Using test_loader as validation
        logging.info(f"Epoch {epoch}/{config.NUM_EPOCHS} - Test Loss: {val_loss:.4f}, Test Metrics: {val_metrics}")

        # Save best model based on AUC
        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_metrics, config.CHECKPOINTS_DIR)
            logging.info(f"New best model saved at Epoch {epoch} with AUC: {best_val_auc:.4f}")

    logging.info(f"\n--- Training Finished ---")
    logging.info(f"Best Test AUC: {best_val_auc:.4f} at Epoch {best_epoch}")

    # --- Final Evaluation on Best Model (optional, can load from checkpoint) ---
    # To ensure you evaluate the model with the best validation performance
    # _, _ = load_checkpoint(model, optimizer, os.path.join(config.CHECKPOINTS_DIR, f"checkpoint_epoch_{best_epoch}.pt"), config.DEVICE)
    # final_test_loss, final_test_metrics = evaluate_model(model, test_loader, criterion, config.DEVICE)
    # logging.info(f"Final evaluation (best checkpoint): Loss: {final_test_loss:.4f}, Metrics: {final_test_metrics}")

def train_val_test_split_dataset(full_dataset, model, optimizer, criterion, config):
    """
    Helper for a standard train/validation/test split on a single dataset,
    useful for initial development or if cross-cohort is not applicable.
    """
    logging.info("Performing standard train/validation/test split.")
    
    # Stratified split to maintain class balance
    labels = [data.y.item() for data in full_dataset]
    
    train_val_indices, test_indices, _, _ = train_test_split(
        range(len(full_dataset)), labels, test_size=(1 - config.TRAIN_TEST_SPLIT_RATIO), 
        stratify=labels, random_state=config.RANDOM_SEED
    )
    
    # Further split train_val into train and validation
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, [labels[i] for i in train_val_indices], test_size=0.2, # 80/20 of the train_val set
        stratify=[labels[i] for i in train_val_indices], random_state=config.RANDOM_SEED
    )

    train_loader = DataLoader(full_dataset[train_indices], batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(full_dataset[val_indices], batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(full_dataset[test_indices], batch_size=config.BATCH_SIZE, shuffle=False)

    logging.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    best_val_auc = -1
    best_epoch = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_metrics = train_model(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, config.DEVICE)
        
        logging.info(f"Epoch {epoch}/{config.NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"  Train Metrics: {train_metrics}")
        logging.info(f"  Val Metrics: {val_metrics}")

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_metrics, config.CHECKPOINTS_DIR)
            logging.info(f"New best model saved at Epoch {epoch} with Val AUC: {best_val_auc:.4f}")

    logging.info(f"\n--- Training Finished for Single Cohort Split ---")
    logging.info(f"Best Validation AUC: {best_val_auc:.4f} at Epoch {best_epoch}")

    # Final evaluation on the test set using the best model
    # To avoid data leakage, load the best model and evaluate on the held-out test set
    _, _ = load_checkpoint(model, optimizer, os.path.join(config.CHECKPOINTS_DIR, f"checkpoint_epoch_{best_epoch}.pt"), config.DEVICE)
    final_test_loss, final_test_metrics = evaluate_model(model, test_loader, criterion, config.DEVICE)
    logging.info(f"Final Test Evaluation (best checkpoint): Loss: {final_test_loss:.4f}, Metrics: {final_test_metrics}")


if __name__ == '__main__':
    # Ensure all directories from config are created
    cfg # This will instantiate Config and create dirs
    
    # Run the main experiment
    run_experiment(cfg)
