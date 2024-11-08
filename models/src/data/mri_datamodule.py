from typing import Any, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.data.components.mri_dataset import *
from src.data.components.custom_batch_sampler import *

class MRIDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            persistent_workers: bool,
            pin_memory: bool,
            basedir: str = '/home/buehlern/Documents/Masterarbeit/data',
            df_name: str = 'df_min',
            max_size_padoutside: int = None,
            pad_to_multiple_of: int = None,
            batch_binning: str = None,
            batch_bins: list[int] = [1152, 1536, 1920, 2304, 2688, 3072],
            normalization_mode: float | str = 'max',
            downsampling: float = None,
            output_channels: int = 1,
            total_data_size: int = None,
            fix_inverted: bool = True,
            train_augmentations: bool = False,
            label: str = 'bodypart',
            stratification_target: str = None,
            val_size: float = 0.05,
            test_size: float = 0.15,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.basedir = basedir
        self.df_name = df_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.max_size_padoutside = max_size_padoutside
        self.pad_to_multiple_of = pad_to_multiple_of
        self.batch_binning = batch_binning
        self.batch_bins = batch_bins
        # normalization_mode (float|'max'|None):
        # None: no normalization is applied (still converted to float32 tensor)
        # float: output is a 0-1-clipped normalization where >= normalization_mode quantile is 1
        self.normalization_mode = normalization_mode
        self.downsampling = downsampling
        # Downsample batch_bins
        if self.downsampling:
            self.batch_bins = [int(bin * self.downsampling) for bin in self.batch_bins]
            print(f"Downsampling batch_bins by factor {self.downsampling} to", self.batch_bins)
        self.output_channels = output_channels
        self.total_data_size = total_data_size
        self.fix_inverted = fix_inverted
        self.train_augmentations = train_augmentations
        self.label = label
        self.stratification_target = stratification_target
        if self.stratification_target is None:
            print(f"Using label {self.label} as stratification_target")
            self.stratification_target = label
        self.val_size = val_size
        self.test_size = test_size

        self.dsbase: torch.utils.data.Dataset = None
        self.data_train: torch.utils.data.Dataset = None
        self.data_val: torch.utils.data.Dataset = None
        self.data_test: torch.utils.data.Dataset = None

        self.setup()

    def prepare_data(self) -> None:
        if not self.dsbase:
            self.dsbase = MRIDatasetBase(
                    basedir = self.basedir,
                    df_name=self.df_name,
                    max_size_padoutside=self.max_size_padoutside,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    pad_to_bins=self.batch_bins,
                    normalization_mode=self.normalization_mode,
                    downsampling=self.downsampling,
                    fix_inverted=self.fix_inverted,
                    label=self.label,
                    output_channels=self.output_channels,
                    total_size=self.total_data_size,
                    stratification_target=self.stratification_target)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            self.prepare_data()

            split_test_loc = self.generate_trainval_test_split()

            print(f"Getting train indices...")
            train_indices = self.get_split_df(split_test_loc, 'train')
            print(f"Done. Train len: {len(train_indices)}")
            print(f"Getting val indices...")
            val_indices = self.get_split_df(split_test_loc, 'val')
            print(f"Done. Val len: {len(val_indices)}")
            print(f"Getting test indices...")
            test_indices = self.get_split_df(split_test_loc, 'test')
            print(f"Done. Test len: {len(test_indices)}")

            print(f"Initializing train dataset...")
            self.data_train = MRIDataset(self.dsbase, train_indices, self.train_augmentations)
            print(f"Done.")
            print(f"Initializing val dataset...")
            self.data_val = MRIDataset(self.dsbase, val_indices, False)
            print(f"Done.")
            print(f"Initializing test dataset...")
            self.data_test = MRIDataset(self.dsbase, test_indices, False)
            print(f"Done.")
    
    def generate_trainval_test_split(self, seed: int = 42):
        # If multiple stratification target values for the same patient, use the rarest one for stratification
        # This computation is always performed globally
        stratification_target_frequencies = self.dsbase.df[self.stratification_target].value_counts()

        size_suffix = f"_size_{self.total_data_size}" if self.total_data_size is not None else ""
        split_test_loc = self.dsbase.basedir / Path(f'splits/split_test_{self.dsbase.df_name}_straton_{self.stratification_target}{size_suffix}.csv')
        if not split_test_loc.exists():
            res = input(
                f'WARN: NO TRAINVAL TEST SPLIT FOUND AT {split_test_loc}, type YES[enter] to generate one: ')
            if res.strip() != 'YES':
                exit(1)

            print('WARN: GENERATING NEW TRAINVAL TEST SPLIT')
            patientid_to_strattarget = {patientid: sorted(set(subdf[self.stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in self.dsbase.df.groupby('patientid')}
            
            _, test = train_test_split(list(patientid_to_strattarget.keys()),
                                       test_size=self.test_size, stratify=list(patientid_to_strattarget.values()), random_state=seed)
            test_patientids = pd.DataFrame(test)
            test_patientids.to_csv(split_test_loc)

        #test_patientids = pd.read_csv(split_test_loc, index_col=0)

        return split_test_loc

    def get_split_df(self, split_test_loc: str, mode: str):
        if not mode in ['train', 'val', 'test', 'train+val', 'train+val+test']:
            raise ValueError(f"Invalid MRIDataset mode: {mode}")
        modeset = set(mode.split('+'))

        # If multiple stratification target values for the same patient, use the rarest one for stratification
        # This computation is always performed globally
        stratification_target_frequencies = self.dsbase.df[self.stratification_target].value_counts()

        df = self.dsbase.df

        test_patientids = pd.read_csv(split_test_loc, index_col=0)

        patientid_index_df = df.set_index('patientid')
        assert set(patientid_index_df.index).issuperset(test_patientids['0'])
        test_idxs = patientid_index_df.loc[test_patientids['0']]['dsbase_index']

        if 'test' in modeset:
            print('WARN: Including test data')
            if modeset == {'test'}:
                # remove trainval
                df = df.loc[test_idxs]
                assert len(set(df['patientid']) - set(test_patientids['0'])) == 0
            assert len(set(df['patientid']) & set(test_patientids['0'])) == len(test_patientids)
        else:
            for idx in test_idxs:
                if idx not in df.index:
                    print(idx, "not in index!")
            df = df.drop(test_idxs)
            assert len(set(df['patientid']) & set(test_patientids['0'])) == 0

        if ('train' in modeset or 'val' in modeset) and not ('train' in modeset and 'val' in modeset):
            patientid_to_strattarget = {patientid: sorted(set(subdf[self.stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in df.groupby('patientid')}
                
            train, val = train_test_split(list(patientid_to_strattarget.keys()),
                                          test_size=self.val_size, stratify=list(patientid_to_strattarget.values()), random_state=42)
            train_patientids = pd.DataFrame(train).rename(columns={0: '0'})
            val_patientids = pd.DataFrame(val).rename(columns={0: '0'})
            val_idxs = patientid_index_df.loc[val_patientids['0']]['dsbase_index']
            if 'val' in modeset:
                # since not both, only keep the val ones
                df = df.loc[val_idxs]
                assert len(set(df['patientid']) & set(val_patientids['0'])) == len(val_patientids)
                assert len(set(df['patientid']) - set(val_patientids['0'])) == 0
            else:
                df = df.drop(val_idxs)
                assert len(set(df['patientid']) & set(train_patientids['0'])) == len(train_patientids)
                assert len(set(df['patientid']) & set(val_patientids['0'])) == 0
        
        return df['dsbase_index']

    def get_batch_sampler(self, data, mode=None):
        if self.batch_binning is not None:
            # Build pixelarr_shapes for faster processing in CustomBatchSampler
            pixelarr_shapes = self.dsbase.df['pixelarr_shape']
            pixelarr_shapes = pixelarr_shapes.iloc[data.indices]
            pixelarr_shapes = pixelarr_shapes.reset_index(drop=True)
            # Downsample pixelarr_shapes dict as well
            if self.downsampling:
                pixelarr_shapes = pixelarr_shapes.apply(lambda shape: (int(shape[0]*self.downsampling), int(shape[1]*self.downsampling)))
            return CustomBatchSampler(
                data, batch_size=self.batch_size, mode=mode,
                binning_strategy=self.batch_binning, bins=self.batch_bins,
                pixelarr_shapes=pixelarr_shapes, prepend_max_size_batch=True,
                drop_last=False)
        else:
            return None

    def collate(self, batch):
        print("Collate", [item[0].shape for item in batch])
        return torch.utils.data.default_collate(batch)

    def train_dataloader(self) -> DataLoader[Any]:
        if self.batch_binning is not None:
            dl = DataLoader(
                dataset=self.data_train,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                batch_sampler=self.get_batch_sampler(self.data_train, mode='train'),
                #shuffle=True,
                #batch_size=self.batch_size
                #collate_fn=self.collate,
            )
            print("DataLoader length", len(dl))
            return dl
        else:
            dl = DataLoader(
                dataset=self.data_train,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                #batch_sampler=self.get_batch_sampler(self.data_train, mode='train'),
                shuffle=True,
                batch_size=self.batch_size,
                #collate_fn=self.collate,
            )
            print("DataLoader length", len(dl))
            return dl

    def val_dataloader(self) -> DataLoader[Any]:
        if self.batch_binning is not None:
            return DataLoader(
                dataset=self.data_val,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                batch_sampler=self.get_batch_sampler(self.data_val, mode='val'),
                #shuffle=False,
                #batch_size=self.batch_size
                #collate_fn=self.collate,
            )
        else:
            return DataLoader(
                dataset=self.data_val,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                #batch_sampler=self.get_batch_sampler(self.data_val),
                shuffle=False,
                batch_size=self.batch_size,
                #collate_fn=self.collate,
            )

    def test_dataloader(self) -> DataLoader[Any]:
        if self.batch_binning is not None:
            return DataLoader(
                dataset=self.data_test,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                batch_sampler=self.get_batch_sampler(self.data_test, mode='test'),
                #shuffle=False,
                #batch_size=self.batch_size
                #collate_fn=self.collate,
            )
        else:
            return DataLoader(
                dataset=self.data_test,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                #batch_sampler=self.get_batch_sampler(self.data_test),
                shuffle=False,
                batch_size=self.batch_size,
                #collate_fn=self.collate,
            )