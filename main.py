import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import os
import argparse

from slaw import SLAW
from dataset import get_dataloaders
from utils import set_seed, History

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 10

def get_model() -> nn.Module:
    """
    initializes a ResNet-18 model pretrained on ImageNet, with its final
    layer adapted for CIFAR-10 classification (10 classes).

    :return: The initialized model, moved to the appropriate device (GPU or CPU).
    :rtype: nn.Module
    """
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

def train_one_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                    criterion, optimizer: optim.Optimizer, run_name: str, 
                    history: History):
    """
    performs one full epoch of training.

    :param model: The neural network model to train.
    :type model: nn.Module
    :param dataloader: The DataLoader for the training data.
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: The loss function (e.g., nn.CrossEntropyLoss or SLAW).
    :param optimizer: The optimization algorithm.
    :type optimizer: optim.Optimizer
    :param run_name: The name of the current experiment run for logging.
    :type run_name: str
    :param history: The History object to log metrics.
    :type history: History
    :return: A tuple of (average training loss, training accuracy).
    :rtype: tuple[float, float]
    """
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        slaw_metrics = {}
        if isinstance(criterion, SLAW):
            loss, slaw_metrics = criterion(outputs, targets)
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)
        if slaw_metrics:
            history.add_step_log(run_name, slaw_metrics)
        progress_bar.set_postfix(loss=total_loss/total_samples, acc=100.*total_correct/total_samples)
    return total_loss / total_samples, 100. * total_correct / total_samples

def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[float, float]:
    """
    evaluates the model on a given dataset.

    :param model: The model to evaluate.
    :type model: nn.Module
    :param dataloader: The DataLoader for the validation or test data.
    :type dataloader: torch.utils.data.DataLoader
    :return: A tuple of (average loss, accuracy).
    :rtype: tuple[float, float]
    """
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
    return total_loss / total_samples, 100. * total_correct / total_samples

def main(args: argparse.Namespace):
    """
    the main driver for running experiments.

    this function parses command-line arguments to configure and run a specific
    training experiment. It handles data loading, model and criterion setup,
    the main training loop, and saving results.

    :param args: An object containing the parsed command-line arguments.
    :type args: argparse.Namespace
    """
    set_seed(42)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    print(f"Using device: {DEVICE}")

    # data loading 
    trainloader, testloader = get_dataloaders(noise_level=args.noise, batch_size=128)

    # exp configrations
    run_name = args.run
    slaw_config_clean = {'gamma': 0.7}
    slaw_config_noisy = {'gamma': 0.9}
    
    model = get_model()

    if run_name == 'baseline':
        criterion = nn.CrossEntropyLoss()
    elif run_name == 'slaw':
        config = slaw_config_noisy if args.noise > 0 else slaw_config_clean
        criterion = SLAW(model, NUM_CLASSES, use_sals=True, use_law=True, **config)
    elif run_name == 'sals_only':
        criterion = SLAW(model, NUM_CLASSES, use_sals=True, use_law=False)
    elif run_name == 'law_only':
        config = slaw_config_noisy if args.noise > 0 else slaw_config_clean
        criterion = SLAW(model, NUM_CLASSES, use_sals=False, use_law=True, **config)
    else:
        raise ValueError(f"Run name '{run_name}' not recognized.")

    # standardize run name for history file
    full_run_name = f"{run_name}_noise{int(args.noise*100)}"
    history_file = f'results/history_noise_{int(args.noise*100)}.pkl'
    
    try:
        history = History()
        history.load(history_file)
        print(f"Loaded existing history from {history_file}")
    except FileNotFoundError:
        history = History()
        print(f"No existing history found. Starting a new one at {history_file}.")
        
    if full_run_name in history.history and len(history.get_epoch_df(full_run_name)) >= N_EPOCHS:
        print(f"--> Skipping '{full_run_name}' as it's already completed.")
        return

    print(f"\n{'='*20} Starting Run: {full_run_name} {'='*20}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    
    history.new_run(full_run_name)
    best_val_acc = 0.0
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, full_run_name, history)
        val_loss, val_acc = evaluate(model, testloader)
        scheduler.step()

        log = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
        history.add_epoch_log(full_run_name, log)
        
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f"results/{full_run_name}_best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best model saved to {model_path} (Acc: {best_val_acc:.2f}%)\n")

    history.save(history_file)
    print(f"Finished run {full_run_name}. Best Val Acc: {best_val_acc:.2f}%. History saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SLAW experiments.')
    parser.add_argument('--run', type=str, required=True, choices=['baseline', 'slaw', 'sals_only', 'law_only'], help='The type of model to run.')
    parser.add_argument('--noise', type=float, default=0.0, help='Symmetric noise level for training data (e.g., 0.2 for 20%).')
    args = parser.parse_args()
    main(args)
