# adapted from https://github.com/Aswathi-Varma/varivit/blob/main/k_fold_training/K_fold_varivit_brats_cbs.py
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, mode, binning_strategy, bins=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.mode = mode
        self.binning_strategy = binning_strategy
        self.bins = bins # Only used in combination with 'smart' binning strategy

        # Create a dictionary that maps image sizes to their indices
        self.size_to_indices = defaultdict(list)
        for idx, (image, _) in enumerate(self.data_source):
            size = image.size()
            # Drop channel information
            size = size[-2:]
            # Consider binning strategy
            if self.binning_strategy == 'smart':
                # Smart binning: Sort image into bin of next bigger size
                bin_size = self.pad_size_to_next_bin(size)
            elif self.binning_strategy == 'strict':
                # Strict binning: Every size forms its own bin
                bin_size = size
            self.size_to_indices[bin_size].append(idx)
        
        self.shuffle_batches()

    def pad_size_to_next_bin(self, size):
        width, height = size
        new_w = next((w for w in self.bins if w >= width), width)
        new_h = next((h for h in self.bins if h >= height), height)
        return (new_w, new_h)

    def shuffle_batches(self):

        self.batches = []

        # Shuffle the indices of each size
        if self.mode == 'train':
            for size in self.size_to_indices:
                np.random.shuffle(self.size_to_indices[size])
        # Create batches based on the dictionary of sizes and indices
        for indices in self.size_to_indices.values():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                # Similar to 'drop last' in the PyTorch DataLoader
                if self.mode == 'train' and len(batch) < self.batch_size:
                    continue

                self.batches.append(batch)

        if self.mode == 'train':
            # Shuffle the batches if in 'train' mode
            np.random.shuffle(self.batches)

    def __iter__(self):
        self.shuffle_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)