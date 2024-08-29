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
            df_name: str = 'clean_df_slim_frac',
            batch_size: int = 64,
            num_workers: int = 0,
            persistent_workers: bool = False,
            pin_memory: bool = False,
            image_size: int = None, # resnet50: 224, ViT: 384
            square: bool = False, # square images
            pad_to_multiple_of: int = None, # typically patch_size
            output_channels: int = 1,
            cache=True,
            total_data_size:int = None,
            batch_binning: str = None,
            batch_bins: list[int] = None,
            fix_inverted: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.df_name = df_name
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.square = square
        self.pad_to_multiple_of = pad_to_multiple_of
        self.output_channels = output_channels
        self.cache = cache
        self.total_data_size = total_data_size
        self.batch_binning = batch_binning
        self.batch_bins = batch_bins
        self.fix_inverted = fix_inverted

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None

        self.setup()

    def prepare_data(self) -> None:
        self.dsbase = MRIDatasetBase(required_cols=None, size=self.image_size, max_size_padoutside=self.image_size, square=self.square, pad_to_multiple_of=self.pad_to_multiple_of, pad_to_bins=self.batch_bins, output_channels=self.output_channels, df_name=self.df_name, cache=self.cache, total_size=self.total_data_size, fix_inverted=self.fix_inverted)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            self.prepare_data()

            self.data_train = MRIDataset(self.dsbase, 'train', seed=0, total_size=self.total_data_size)
            self.data_val = MRIDataset(self.dsbase, 'val', seed=0, total_size=self.total_data_size)
            self.data_test = MRIDataset(self.dsbase, 'test', seed=0, total_size=self.total_data_size)
    
    def get_batch_sampler(self, data, mode=None):
        if self.batch_binning is not None:
            # Build img_size_map for faster processing in CustomBatchSampler
            img_size_map = data.df[['dsbase_index', 'pixelarr_shape']]
            img_size_map = img_size_map.loc[:, ~img_size_map.columns.duplicated()]
            return CustomBatchSampler(data, batch_size=self.batch_size_per_device, mode=mode, binning_strategy=self.batch_binning, bins=self.batch_bins, img_size_map=img_size_map)
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
                #batch_size=self.batch_size_per_device
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
                batch_size=self.batch_size_per_device
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
                #batch_size=self.batch_size_per_device
            )
        else:
            return DataLoader(
                dataset=self.data_val,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                #batch_sampler=self.get_batch_sampler(self.data_val),
                shuffle=False,
                batch_size=self.batch_size_per_device
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
                #batch_size=self.batch_size_per_device
            )
        else:
            return DataLoader(
                dataset=self.data_test,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                #batch_sampler=self.get_batch_sampler(self.data_test),
                shuffle=False,
                batch_size=self.batch_size_per_device
            )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass