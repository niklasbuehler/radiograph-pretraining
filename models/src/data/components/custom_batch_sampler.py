# adapted from https://github.com/Aswathi-Varma/varivit/blob/main/k_fold_training/K_fold_varivit_brats_cbs.py
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, mode):
        self.data_source = data_source
        self.batch_size = batch_size
        self.mode = mode

        # Create a dictionary that maps image sizes to their indices
        self.size_to_indices = defaultdict(list)
        for idx, (image, _) in enumerate(self.data_source):
            size = image.size()
            self.size_to_indices[size].append(idx)
        
        self.shuffle_batches()

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

                #Similar to 'drop last' in the PyTorch DataLoader
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