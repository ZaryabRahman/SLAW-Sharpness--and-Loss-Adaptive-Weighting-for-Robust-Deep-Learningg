import torch
import numpy as np
import pickle
import pandas as pd
import torch.nn.functional as F

def set_seed(seed=42):
    """For reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class History:
    """A helper class to store and manage training history."""
    def __init__(self): self.history = {}
    def new_run(self, name): self.history[name] = {'epochs': [], 'steps': []}
    def add_epoch_log(self, name, log): self.history[name]['epochs'].append(log)
    def add_step_log(self, name, log): self.history[name]['steps'].append(log)
    def get_epoch_df(self, name): return pd.DataFrame(self.history[name]['epochs'])
    def get_step_df(self, name): return pd.DataFrame(self.history[name]['steps'])
    def save(self, fp):
        with open(fp, 'wb') as f: pickle.dump(self.history, f)
    def load(self, fp):
        with open(fp, 'rb') as f: self.history = pickle.load(f)

@torch.no_grad()
def calculate_ece(model, dataloader, n_bins=15, device='cuda'):
    """Calculates the Expected Calibration Error of a model."""
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
def get_calibration_data(model, dataloader, n_bins=15, device='cuda'):
    """Gathers data needed for plotting a reliability diagram."""
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
