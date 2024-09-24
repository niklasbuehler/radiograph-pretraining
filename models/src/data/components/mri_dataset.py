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
        df = df.reset_index(drop=True)

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

        # Padding and resizing
        if self.pad_to_bins is not None:
            # Pad the image to next bin size
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
        curitem_series = self.df.loc[index]
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
            mode: str,
            stratification_target: str,
            total_size: int,
            val_size: float = 0.05,
            test_size: float = 0.15,
            seed: int = 42):
        super().__init__()
        self.dsbase = dsbase
        self.mode = mode
        self.stratification_target = stratification_target
        self.seed = seed
        self.val_size = val_size
        self.test_size = test_size
        self.total_size = total_size

        print(f"Initializing MRIDataset(mode={self.mode})...")

        if not mode in ['train', 'val', 'test', 'train+val', 'train+val+test']:
            raise ValueError(f"Invalid MRIDataset mode: {mode}")
        modeset = set(mode.split('+'))
        self.modeset = modeset

        self.df = dsbase.df.copy()
        self.df.reset_index(names='dsbase_index', inplace=True)

        # If multiple stratification target values for the same patient, use the rarest one for stratification
        # This computation is always performed globally
        stratification_target_frequencies = dsbase.df[stratification_target].value_counts()

        size_suffix = f"_size_{self.total_size}" if self.total_size is not None else ""
        split_test_loc = self.dsbase.basedir / Path(f'splits/split_test_{self.dsbase.df_name}_straton_{stratification_target}{size_suffix}.csv')
        if not split_test_loc.exists():
            res = input(
                f'WARN: NO TRAINVAL TEST SPLIT FOUND AT {split_test_loc}, type YES[enter] to generate one: ')
            if res.strip() != 'YES':
                self.df = None
                exit(1)

            print('WARN: GENERATING NEW TRAINVAL TEST SPLIT')
            patientid_to_strattarget = {patientid: sorted(set(subdf[stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in self.df.groupby('patientid')}

            _, test = train_test_split(list(patientid_to_strattarget.keys()),
                                       test_size=test_size, stratify=list(patientid_to_strattarget.values()), random_state=0)
            test_patientids = pd.DataFrame(test)
            test_patientids.to_csv(split_test_loc)
        test_patientids = pd.read_csv(split_test_loc, index_col=0)

        patientid_index_df = self.df.set_index('patientid')
        assert set(patientid_index_df.index).issuperset(test_patientids['0'])
        test_idxs = patientid_index_df.loc[test_patientids['0']]['dsbase_index']
        if 'test' in modeset:
            print('WARN: Including test data')
            if modeset == {'test'}:
                # remove trainval
                self.df = self.df.loc[test_idxs]
                assert len(set(self.df['patientid']) - set(test_patientids['0'])) == 0
            assert len(set(self.df['patientid']) & set(test_patientids['0'])) == len(test_patientids)
        else:
            for idx in test_idxs:
                if idx not in self.df.index:
                    print(idx, "not in index!")
            self.df = self.df.drop(test_idxs)
            assert len(set(self.df['patientid']) & set(test_patientids['0'])) == 0

        patientid_index_df = self.df.set_index('patientid')
        if ('train' in modeset or 'val' in modeset) and not ('train' in modeset and 'val' in modeset):
            patientid_to_strattarget = {patientid: sorted(set(subdf[stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in self.df.groupby('patientid')}
                
            train, val = train_test_split(list(patientid_to_strattarget.keys()),
                                          test_size=val_size, stratify=list(patientid_to_strattarget.values()), random_state=seed)
            train_patientids = pd.DataFrame(train).rename(columns={0: '0'})
            val_patientids = pd.DataFrame(val).rename(columns={0: '0'})
            val_idxs = patientid_index_df.loc[val_patientids['0']]['dsbase_index']
            if 'val' in modeset:
                # since not both, only keep the val ones
                self.df = self.df.loc[val_idxs]
                assert len(set(self.df['patientid']) & set(val_patientids['0'])) == len(val_patientids)
                assert len(set(self.df['patientid']) - set(val_patientids['0'])) == 0
            else:
                self.df = self.df.drop(val_idxs)
                assert len(set(self.df['patientid']) & set(train_patientids['0'])) == len(train_patientids)
                assert len(set(self.df['patientid']) & set(val_patientids['0'])) == 0
        
        print(self, 'initialized')

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        return f'MRIDataset(mode={self.mode}, len={len(self.df)})'

    def __str__(self) -> str:
        return repr(self)

    def __getitem__(self, index):
        return self.dsbase[self.df.iloc[index]['dsbase_index']]

    def getrow(self, index):
        return self.df.iloc[index]