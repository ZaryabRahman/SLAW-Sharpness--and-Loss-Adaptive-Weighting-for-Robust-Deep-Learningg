import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class NoisyCIFAR10(Dataset):
    """A wrapper for CIFAR-10 that introduces symmetric label noise."""
    def __init__(self, cifar_dataset, noise_rate=0.2, random_state=42):
        self.dataset = cifar_dataset
        self.num_classes = len(cifar_dataset.classes)
        self.noise_rate = noise_rate
        self.original_targets = np.array(cifar_dataset.targets)
        self.noisy_targets = self._create_noisy_labels(random_state)

        # replace the dataset's targets with our noisy ones
        self.dataset.targets = self.noisy_targets

        print(f"created a NoisyCIFAR10 dataset with {noise_rate*100}% symmetric noise.")
        correct_count = np.sum(self.original_targets == self.noisy_targets)
        print(f"actual agreement with original labels: {correct_count / len(self.original_targets):.2%}\n")

    def _create_noisy_labels(self, random_state):
        rng = np.random.RandomState(random_state)
        noisy_targets = self.original_targets.copy()
        num_samples = len(self.original_targets)
        num_to_corrupt = int(self.noise_rate * num_samples)
        indices_to_corrupt = rng.choice(num_samples, num_to_corrupt, replace=False)

        for idx in indices_to_corrupt:
            original_label = noisy_targets[idx]
            potential_new_labels = [l for l in range(self.num_classes) if l != original_label]
            new_label = rng.choice(potential_new_labels)
            noisy_targets[idx] = new_label
        return list(noisy_targets)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

def get_dataloaders(noise_level=0.0, batch_size=128):
    print("preparing CIFAR-10 data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    if noise_level > 0:
        trainset = NoisyCIFAR10(trainset_clean, noise_rate=noise_level)
    else:
        trainset = trainset_clean

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader
