import numpy as np
import torch
import random
import os
import sys
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from typing import Dict, Any, Union

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility.")

def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], y_pred_proba: Union[np.ndarray, torch.Tensor], threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculates various classification metrics.

    Args:
        y_true (Union[np.ndarray, torch.Tensor]): True labels (binary).
        y_pred_proba (Union[np.ndarray, torch.Tensor]): Predicted probabilities.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        Dict[str, float]: A dictionary containing calculated metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = y_pred_proba.cpu().numpy()

    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "AUC": roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else float('nan'),
        "Accuracy": accuracy_score(y_true, y_pred_binary),
        "F1_score": f1_score(y_true, y_pred_binary),
        "Precision": precision_score(y_true, y_pred_binary),
        "Recall": recall_score(y_true, y_pred_binary)
    }
    return metrics

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float], save_path: str) -> None:
    """
    Saves the model and optimizer state.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        metrics (Dict[str, float]): Dictionary of current evaluation metrics.
        save_path (str): Directory to save the checkpoint.
    """
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, device: torch.device) -> Tuple[int, Dict[str, float]]:
    """
    Loads model and optimizer state from a checkpoint.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device to load the checkpoint onto.

    Returns:
        Tuple[int, Dict[str, float]]: Loaded epoch number and metrics.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        return 0, {} # Return default if not found

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    print(f"Checkpoint loaded from {checkpoint_path} (Epoch: {epoch}, Metrics: {metrics})")
    return epoch, metrics

def setup_logging(log_file: str) -> None:
    """
    Sets up basic logging to both console and a file.

    Args:
        log_file (str): Path to the log file.
    """
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Add the file handler if not already added
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file) for handler in root_logger.handlers):
        root_logger.addHandler(file_handler)

    # Add a console handler if not already added
    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    print(f"Logging set up. Output directed to console and {log_file}")
    import logging # Import logging after setup for demonstration
    logging.info("Logging initialized.")
