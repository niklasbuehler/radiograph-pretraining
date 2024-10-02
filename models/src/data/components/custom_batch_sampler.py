# adapted from https://github.com/Aswathi-Varma/varivit/blob/main/k_fold_training/K_fold_varivit_brats_cbs.py
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, mode, binning_strategy, bins=None, pixelarr_shapes=None, prepend_max_size_batch=False, drop_last=False):
        self.data_source = data_source
         # use pixelarr_shapes extracted from dataframe to avoid having to open every image for filling the bins 
        self.pixelarr_shapes = pixelarr_shapes
        self.batch_size = batch_size
        self.mode = mode
        self.binning_strategy = binning_strategy
        self.bins = sorted(bins) # Only used in combination with 'smart' binning strategy
        self.prepend_max_size_batch = prepend_max_size_batch
        self.drop_last = drop_last

        # Create a dict that maps image shapes to indices of images with shape
        print(f"Generating shape_to_indices dict in CustomBatchSampler...")
        self.shape_to_indices = defaultdict(list)
        if self.pixelarr_shapes is not None:
            # Use pixelarr_shapes if available to avoid loading all images
            for i, shape in enumerate(self.pixelarr_shapes):
                # Consider binning strategy
                if self.binning_strategy == 'smart':
                    # Smart binning: Sort image into bin of next bigger shape
                    bin_shape = self.pad_shape_to_next_bin(shape)
                elif self.binning_strategy == 'strict':
                    # Strict binning: Every shape forms its own bin
                    bin_shape = shape
                #print("Storing index", i, "of image with shape", shape, "in bin", bin_shape)
                self.shape_to_indices[bin_shape].append(i)
        else:
            print("WARN: No pixelarr_shapes dict available")
            for i, (image, _) in enumerate(self.data_source):
                shape = image.size()
                # Drop channel information
                shape = shape[-2:]
                # Consider binning strategy
                if self.binning_strategy == 'smart':
                    # Smart binning: Sort image into bin of next bigger shape
                    bin_shape = self.pad_shape_to_next_bin(shape)
                elif self.binning_strategy == 'strict':
                    # Strict binning: Every shape forms its own bin
                    bin_shape = shape
                self.shape_to_indices[bin_shape].append(i)
        print(f"Done.")
        
        self.shuffle_batches()

    def pad_shape_to_next_bin(self, shape):
        width, height = shape
        new_w = next((w for w in self.bins if w >= width), width)
        new_h = next((h for h in self.bins if h >= height), height)
        return (new_w, new_h)

    def shuffle_batches(self):

        self.batches = []

        # Shuffle the indices of each shape
        if self.mode == 'train':
            for shape in self.shape_to_indices:
                np.random.shuffle(self.shape_to_indices[shape])
        
        # Find bin of largest size that contains a full batch
        max_size_batch = None
        if self.prepend_max_size_batch:
            bin_shapes = sorted(self.shape_to_indices.keys(), reverse=True)
            max_shape = None
            min_count = self.batch_size if self.mode == 'train' else 1
            for shape in bin_shapes:
                if len(self.shape_to_indices[shape]) >= min_count and max_shape is None:
                    max_shape = shape
                    break
            print(f"Maximum bin shape: {max_shape}")
            # Single out a batch of maximum shape
            max_size_indices = self.shape_to_indices[max_shape]
            max_size_batch = max_size_indices[:self.batch_size]
            self.shape_to_indices[max_shape] = max_size_indices[self.batch_size:]

        # Create batches based on the dictionary of sizes and indices
        for indices in self.shape_to_indices.values():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                # Similar to 'drop last' in the PyTorch DataLoader
                if self.mode == 'train' and len(batch) < self.batch_size and self.drop_last:
                    continue
                
                self.batches.append(batch)
        
        if self.mode == 'train':
            # Shuffle the batches if in 'train' mode
            np.random.shuffle(self.batches)
        
        # Add max_size_batch as the first batch
        if self.prepend_max_size_batch:
            self.batches.insert(0, max_size_batch)

    def __iter__(self):
        self.shuffle_batches()
        for batch in self.batches:
            #print("Batch", batch, "with shapes",
            #        [self.pixelarr_shapes.iloc[b_idx] for b_idx in batch],
            #        "->", [self.pad_shape_to_next_bin(self.pixelarr_shapes.iloc[b_idx]) for b_idx in batch])
            yield batch

    def __len__(self):
        return len(self.batches)