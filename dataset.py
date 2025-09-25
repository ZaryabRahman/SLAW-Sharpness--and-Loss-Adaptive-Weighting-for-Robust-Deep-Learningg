import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class NoisyCIFAR10(Dataset):
    """
    a dataset wrapper for CIFAR-10 that introduces symmetric label noise.

    this class takes a standard CIFAR-10 dataset object and corrupts a specified
    fraction of its labels. For each corrupted sample, the true label is replaced
    with a new label chosen uniformly at random from the other classes.

    :param cifar_dataset: An instance of the CIFAR-10 dataset (e.g., from torchvision).
    :type cifar_dataset: torch.utils.data.Dataset
    :param noise_rate: The fraction of labels to corrupt (e.g., 0.2 for 20% noise).
    :type noise_rate: float
    :param random_state: A seed for the random number generator to ensure
                         reproducible noise generation.
    :type random_state: int
    """
    def __init__(self, cifar_dataset: Dataset, noise_rate: float = 0.2, random_state: int = 42):
        self.dataset = cifar_dataset
        self.num_classes = len(cifar_dataset.classes)
        self.noise_rate = noise_rate
        self.original_targets = np.array(cifar_dataset.targets)
        self.noisy_targets = self._create_noisy_labels(random_state)

        # overwrite the dataset's targets with our noisy ones
        self.dataset.targets = self.noisy_targets

        print(f"Created a NoisyCIFAR10 dataset with {noise_rate*100}% symmetric noise.")
        correct_count = np.sum(self.original_targets == self.noisy_targets)
        print(f"Actual agreement with original labels: {correct_count / len(self.original_targets):.2%}\n")

    def _create_noisy_labels(self, random_state: int):
        """
        generates noisy labels by symmetrically corrupting the original labels.

        :param random_state: Seed for the random number generator.
        :type random_state: int
        :return: A list of potentially noisy labels.
        :rtype: list
        """
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

    def __getitem__(self, index: int):
        """
        retrieves an item from the dataset.

        delegates to the underlying dataset's `__getitem__`, which will now use
        the corrupted `self.dataset.targets`.

        :param index: The index of the item.
        :type index: int
        :return: A tuple of (image, noisy_label).
        :rtype: tuple
        """
        return self.dataset[index]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: The size of the dataset.
        :rtype: int
        """
        return len(self.dataset)

def get_dataloaders(noise_level: float = 0.0, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    creates and returns CIFAR-10 train and test DataLoaders.

    if `noise_level` is greater than 0, it applies symmetric label noise to the
    training dataset using the `NoisyCIFAR10` wrapper.

    :param noise_level: The fraction of training labels to corrupt. Defaults to 0.0 (no noise).
    :type noise_level: float
    :param batch_size: The batch size for the DataLoaders. Defaults to 128.
    :type batch_size: int
    :return: A tuple containing the training DataLoader and the test DataLoader.
    :rtype: tuple[DataLoader, DataLoader]
    """
    print("Preparing CIFAR-10 data...")
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
        print("Using clean training data.\n")
        trainset = trainset_clean

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader
