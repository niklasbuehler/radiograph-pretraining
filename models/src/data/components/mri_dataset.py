import torch
import numpy as np
import pandas as pd
import pydicom
import functools
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class MRIDatasetBase(torch.utils.data.Dataset):
    def __init__(
            self,
            basedir: str,
            df_name: str,
            max_size_padoutside: int,
            pad_to_multiple_of: int,
            pad_to_bins: list[int],
            normalization_mode: float | str,
            downsampling: float,
            fix_inverted: bool,
            label: str,
            output_channels: int,
            total_size: int,
            stratification_target: str):
        
        super().__init__()
        
        # Directory of the dataframe
        self.basedir = Path(basedir)
        # Name of the dataframe pickle file
        self.df_name = df_name
        
        # Size parameters
        # max_size_padoutside: Pad to square of shape (max_size, max_size)
        # pad_to_multiple_of: Pad both sides (independently) to multiples of given value
        # pad_to_bins: Pad both sides (independently) to the next biggest size from given list of size bins
        self.max_size_padoutside = max_size_padoutside
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_bins = pad_to_bins

        # normalization_mode (float|'max'|None):
        # None: no normalization is applied (still converted to float32 tensor)
        # float: output is a 0-1-clipped normalization where >= normalization_mode quantile is 1
        self.normalization_mode = normalization_mode

        # Whether or not to downsample every image
        self.downsampling = downsampling
        
        # Whether or not the inverted flag in the dataframe should be used to fix inverted images
        self.fix_inverted = fix_inverted

        # Label column from dataframe to return
        self.label = label

        # Number of output channels
        self.output_channels = output_channels
        
        # Total size to which the dataset should be limited
        self.total_size = total_size

        # Stratification target
        self.stratification_target = stratification_target

        print("Initializing MRIDatasetBase...")

        df_path = self.basedir / f"{self.df_name}.pkl"
        print(f"Loading dataframe from {df_path}...")
        df = pd.read_pickle(df_path)

        # Optional: Limiting total size of dataset (including stratification)
        if self.total_size is not None:
            size_per_strat_val = int(self.total_size / len(df[self.stratification_target].unique()))
            print(f"Limiting dataset total size to {self.total_size}")
            print(f"Size for each {self.stratification_target}: {size_per_strat_val}")
            df = df.groupby(self.stratification_target, group_keys=False).apply(lambda x: x.sample(min(len(x), size_per_strat_val)))
            print("New size of dataset:", len(df))
        
        df = df.dropna(subset=[self.label])
        
        df = df.sort_values('path')
        #df = df.reset_index(drop=True)
        df['dsbase_index'] = df.index

        self.label_to_idx = {
            lbl: i for i, lbl in enumerate(sorted(set(df[label])))
        }
        self.idx_to_bodypart = {
            i: lbl for i, lbl in enumerate(sorted(set(df[label])))
        }

        self.df = df

        print(self, 'initialized')

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        return f'MRIDatasetBase(len={len(self.df)})'

    def __str__(self) -> str:
        return repr(self)

    @functools.lru_cache
    def _getitem_innercached(self, index):
        return self._getitem_inner(index)

    def _getpixelarray(self, curitem_series):
        try:
            dcm = pydicom.dcmread(curitem_series['path'])
        except (AttributeError, OSError):
            null_pixel_array = np.ones((1, 1)) * np.nan
            return torch.tensor(null_pixel_array, dtype=torch.float32)[None]

        pixel_array = dcm.pixel_array

        # Add batch dim
        pixel_array = torch.tensor(pixel_array, dtype=torch.float32)[None]

        # Normalize
        if self.normalization_mode == None:
            pixel_array = pixel_array.to(torch.float32)
        elif self.normalization_mode == 'max':
            pixel_array /= pixel_array.max()
        elif isinstance(self.normalization_mode, float) or isinstance(self.normalization_mode, int):
                pixel_array = pixel_array.astype(float)
                pixel_array /= np.quantile(pixel_array, self.normalization_mode)
                pixel_array = np.clip(pixel_array, 0, 1)

        # Fix inverted scans
        if self.fix_inverted:
            inverted = curitem_series['inverted']
            if inverted:
                pixel_array = torch.max(pixel_array) - pixel_array

        # Downsampling
        if self.downsampling:
            pixel_array = F.interpolate(pixel_array, scale_factor=self.downsampling)

        # Padding and resizing
        if self.pad_to_bins is not None:
            # Pad the image to next bin size
            old_shape = pixel_array.shape
            height, width = pixel_array.shape[-2:]
            new_width = next((w for w in self.pad_to_bins if w >= width), width)
            new_height = next((h for h in self.pad_to_bins if h >= height), height)
            
            missing_cols = new_width - pixel_array.shape[-1]
            pad_left = missing_cols // 2
            pad_right = sum(divmod(missing_cols, 2))

            missing_rows = new_height - pixel_array.shape[-2]
            pad_top = missing_rows // 2
            pad_bottom = sum(divmod(missing_rows, 2))
            pixel_array = F.pad(pixel_array, [0, pad_left+pad_right, 0, pad_top+pad_bottom])
            #print(f"Padded {old_shape} to {pixel_array.shape}")
        elif self.pad_to_multiple_of is not None:
            # Pad the image to the next bigger multiple of pad_to_multiple_of
            missing_cols = (self.pad_to_multiple_of - pixel_array.shape[-1] % self.pad_to_multiple_of) % self.pad_to_multiple_of
            pad_left = missing_cols // 2
            pad_right = sum(divmod(missing_cols, 2))

            missing_rows = (self.pad_to_multiple_of - pixel_array.shape[-2] % self.pad_to_multiple_of) % self.pad_to_multiple_of
            pad_top = missing_rows // 2
            pad_bottom = sum(divmod(missing_rows, 2))

            pixel_array = F.pad(pixel_array, [0, pad_left+pad_right, 0, pad_top+pad_bottom])
        elif self.max_size_padoutside is not None:
            max_size = self.max_size_padoutside # max(pixel_array.shape)
            missing_cols = max_size - pixel_array.shape[-1]
            pad_left = missing_cols // 2
            pad_right = sum(divmod(missing_cols, 2))

            missing_rows = max_size - pixel_array.shape[-2]
            pad_top = missing_rows // 2
            pad_bottom = sum(divmod(missing_rows, 2))

            pixel_array = F.pad(pixel_array, [0, pad_left+pad_right, 0, pad_top+pad_bottom])
        
        return pixel_array

    def _getitem_inner(self, index):
        curitem_series = self.df.iloc[index]
        pixel_array = self._getpixelarray(curitem_series)
        res = dict(pixel_array=pixel_array, label=self.label_to_idx[curitem_series[self.label]])
        return res

    def __getitem__(self, index):
        item = self._getitem_inner(index)
        return item['pixel_array'].expand(self.output_channels, -1, -1), item['label']



class MRIDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dsbase: MRIDatasetBase,
            indices: pd.DataFrame):
        super().__init__()
        self.dsbase = dsbase
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __repr__(self) -> str:
        return f'MRIDataset(len={len(self.indices)})'

    def __str__(self) -> str:
        return repr(self)

    def __getitem__(self, index):
        #print("Dataset index", index, "->", self.indices.iloc[index])
        return self.dsbase[self.indices.iloc[index]]


    def getrow(self, index):
        return self.indices.iloc[index]