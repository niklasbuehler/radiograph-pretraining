from typing import Any, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.data.components.mri_dataset import *
from src.data.components.custom_batch_sampler import *
#from custom_batch_sampler import *

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
            output_channels: int = 1,
            total_data_size: int = None,
            fix_inverted: bool = True,
            label: str = 'bodypart',
            stratification_target: str = None,
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
        self.output_channels = output_channels
        self.total_data_size = total_data_size
        self.fix_inverted = fix_inverted
        self.label = label
        self.stratification_target = stratification_target
        if self.stratification_target is None:
            print(f"Using label {self.label} as stratification_target")
            self.stratification_target = label

        self.data_train: torch.utils.data.Dataset = None
        self.data_val: torch.utils.data.Dataset = None
        self.data_test: torch.utils.data.Dataset = None

        self.setup()

    def prepare_data(self) -> None:
        self.dsbase = MRIDatasetBase(
                basedir = self.basedir,
                df_name=self.df_name,
                max_size_padoutside=self.max_size_padoutside,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_to_bins=self.batch_bins,
                normalization_mode=self.normalization_mode,
                fix_inverted=self.fix_inverted,
                label=self.label,
                output_channels=self.output_channels,
                total_size=self.total_data_size,
                stratification_target=self.stratification_target)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            self.prepare_data()

            self.data_train = MRIDataset(self.dsbase, 'train', stratification_target=self.stratification_target, total_size=self.total_data_size, seed=0)
            self.data_val = MRIDataset(self.dsbase, 'val', stratification_target=self.stratification_target, total_size=self.total_data_size, seed=0)
            self.data_test = MRIDataset(self.dsbase, 'test', stratification_target=self.stratification_target, total_size=self.total_data_size, seed=0)
    
    def get_batch_sampler(self, data, mode=None):
        if self.batch_binning is not None:
            # Build img_size_map for faster processing in CustomBatchSampler
            img_size_map = data.df[['dsbase_index', 'pixelarr_shape']]
            img_size_map = img_size_map.loc[:, ~img_size_map.columns.duplicated()]
            img_size_map.reset_index(inplace=True)
            return CustomBatchSampler(data, batch_size=self.batch_size, mode=mode, binning_strategy=self.batch_binning, bins=self.batch_bins, img_size_map=img_size_map)
        else:
            return None

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
                batch_size=self.batch_size
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
            )
        else:
            return DataLoader(
                dataset=self.data_val,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                #batch_sampler=self.get_batch_sampler(self.data_val),
                shuffle=False,
                batch_size=self.batch_size
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
            )
        else:
            return DataLoader(
                dataset=self.data_test,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                #batch_sampler=self.get_batch_sampler(self.data_test),
                shuffle=False,
                batch_size=self.batch_size
            )