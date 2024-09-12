# adapted from https://github.com/Aswathi-Varma/varivit/blob/main/k_fold_training/K_fold_varivit_brats_cbs.py
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, mode, binning_strategy, bins=None, img_size_map=None, prepend_max_size_batch=False):
        self.data_source = data_source
         # use img_size_map extracted from dataframe to avoid having to open every image for filling the bins 
        self.img_size_map = img_size_map
        self.batch_size = batch_size
        self.mode = mode
        self.binning_strategy = binning_strategy
        self.bins = sorted(bins) # Only used in combination with 'smart' binning strategy
        self.prepend_max_size_batch = prepend_max_size_batch

        # Create a dictionary that maps image sizes to their indices
        self.size_to_indices = defaultdict(list)
        if self.img_size_map is not None:
            # Use img_size_map if available to avoid loading all images
            for idx, iter_row in enumerate(self.img_size_map.iterrows()):
                i, row = iter_row
                size = row['pixelarr_shape']
                # Consider binning strategy
                if self.binning_strategy == 'smart':
                    # Smart binning: Sort image into bin of next bigger size
                    bin_size = self.pad_size_to_next_bin(size)
                elif self.binning_strategy == 'strict':
                    # Strict binning: Every size forms its own bin
                    bin_size = size
                #print("Storing", idx, "with size", size, "in bin", bin_size)
                self.size_to_indices[bin_size].append(idx)
        else:
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
        
        # Find bin of largest size that contains a full batch
        if self.prepend_max_size_batch:
            bin_sizes = sorted(self.size_to_indices.keys(), reverse=True)
            max_size = None
            min_count = self.batch_size if self.mode == 'train' else 1
            for size in bin_sizes:
                if len(self.size_to_indices[size]) >= min_count and max_size is None:
                    max_size = size
            print(f"Maximum bin size: {max_size}")
        

        # Create batches based on the dictionary of sizes and indices
        max_size_batch = None
        for indices in self.size_to_indices.values():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                # Similar to 'drop last' in the PyTorch DataLoader
                if self.mode == 'train' and len(batch) < self.batch_size:
                    continue
                
                # Save one batch of maximum size to put as first batch, in order to avoid recompiling the model
                if self.prepend_max_size_batch and max_size_batch is None:
                    img_sizes = self.img_size_map.loc[self.img_size_map["dsbase_index"] == batch[0]]["pixelarr_shape"]
                    if len(img_sizes) > 0:
                        img_size = img_sizes.values[0]
                        orig_img_size = img_size
                        #print(f"Checking batch with shape {img_size}")
                        if self.binning_strategy == "smart":
                            img_size = self.pad_size_to_next_bin(img_size)
                            #print(f"Padding shape to {img_size}")
                        if img_size == max_size:
                            # Save batch and don't add it to list of batches yet
                            print(f"Max size found: pad({orig_img_size})={img_size} == {max_size}")
                            print(f"First image id {batch[0]}")
                            max_size_batch = batch
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
            yield batch

    def __len__(self):
        return len(self.batches)