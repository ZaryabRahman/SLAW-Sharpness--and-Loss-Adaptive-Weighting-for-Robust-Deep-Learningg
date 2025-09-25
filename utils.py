import torch
import numpy as np
import pickle
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

def set_seed(seed: int = 42):
    """
    sets the random seed for all relevant libraries to ensure reproducibility.

    :param seed: The integer value to use as the seed.
    :type seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

class History:
    """
    a utility class to log, manage, and save training history.

    this class provides a simple interface to track metrics on both a per-epoch
    and per-step basis, and can save/load its state to/from a pickle file.
    """
    def __init__(self):
        self.history = {}

    def new_run(self, name: str):
        """starts a new logging 
        session for an 
        experiment run.
        """
        self.history[name] = {'epochs': [], 'steps': []}

    def add_epoch_log(self, name: str, log: dict):
        """adds a log entry
        for a completed
        epoch."""
        self.history[name]['epochs'].append(log)

    def add_step_log(self, name: str, log: dict):
        """adds a log entry \
        for a completed training 
        step (batch).
        """
        self.history[name]['steps'].append(log)

    def get_epoch_df(self, name: str) -> pd.DataFrame:
        """returns the epoch 
        history for a run as a 
        pandas DataFrame.
        """
        return pd.DataFrame(self.history[name]['epochs'])

    def get_step_df(self, name: str) -> pd.DataFrame:
        """returns the step history 
        for a run as a pandas DataFrame.
        """
        return pd.DataFrame(self.history[name]['steps'])

    def save(self, fp: str):
        """Ssaves the entire history
        dictionary to a file.
        """
        with open(fp, 'wb') as f:
            pickle.dump(self.history, f)

    def load(self, fp: str):
        """Loads a history dictionary from a file."""
        with open(fp, 'rb') as f:
            self.history = pickle.load(f)

@torch.no_grad()
def calculate_ece(model: torch.nn.Module, dataloader: DataLoader, n_bins: int = 15, device: str = 'cuda') -> float:
    """
    calculates the Expected Calibration Error (ECE) of a model.

    ECE is a metric that measures the discrepancy between a model's prediction
    confidence and its actual accuracy. A lower ECE indicates a better-calibrated model.

    :param model: The trained model to evaluate.
    :type model: torch.nn.Module
    :param dataloader: DataLoader for the dataset to evaluate on (e.g., test set).
    :type dataloader: DataLoader
    :param n_bins: The number of confidence bins to use for the calculation.
    :type n_bins: int
    :param device: The device to run the evaluation on ('cuda' or 'cpu').
    :type device: str
    :return: The computed ECE score.
    :rtype: float
    """
    model.eval()
    all_confidences, all_correct = [], []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        correct = predictions.eq(targets)
        all_confidences.append(confidences.cpu())
        all_correct.append(correct.cpu())

    all_confidences = torch.cat(all_confidences)
    all_correct = torch.cat(all_correct)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            accuracy_in_bin = all_correct[in_bin].float().mean()
            avg_confidence_in_bin = all_confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

@torch.no_grad()
def get_calibration_data(model: torch.nn.Module, dataloader: DataLoader, n_bins: int = 15, device: str = 'cuda') -> tuple:
    """
    gathers the necessary statistics to plot a reliability diagram.

    :param model: The trained model to evaluate.
    :type model: torch.nn.Module
    :param dataloader: DataLoader for the dataset.
    :type dataloader: DataLoader
    :param n_bins: The number of confidence bins.
    :type n_bins: int
    :param device: The device for evaluation.
    :type device: str
    :return: A tuple containing lists of (accuracies per bin, average confidences
             per bin, and proportion of samples per bin).
    :rtype: tuple[list, list, list]
    """
    model.eval()
    all_confidences, all_correct = [], []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        correct = predictions.eq(targets)
        all_confidences.append(confidences.cpu())
        all_correct.append(correct.cpu())

    all_confidences = torch.cat(all_confidences)
    all_correct = torch.cat(all_correct)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
    
    accuracies, avg_confs, bin_props = [], [], []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            accuracy_in_bin = all_correct[in_bin].float().mean()
            avg_confidence_in_bin = all_confidences[in_bin].mean()
            accuracies.append(accuracy_in_bin.item())
            avg_confs.append(avg_confidence_in_bin.item())
        else:
            accuracies.append(0)
            avg_confs.append(0)
        bin_props.append(prop_in_bin.item())
    return accuracies, avg_confs, bin_props
