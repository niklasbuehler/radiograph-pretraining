# %%
import functools
import hashlib
import inspect
import itertools
import json
import os
import pickle
import re
import stat
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydicom
import seaborn as sns
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from IPython import get_ipython
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import skimage.io
from tqdm import tqdm

while not Path('.toplevel').exists() and Path('..').resolve() != Path().resolve():
    os.chdir(Path('..'))
if str(Path().resolve()) not in sys.path:
    sys.path.insert(0, str(Path().resolve()))

import src.data.components.helpers as helpers

class IDPDatasetBase(torch.utils.data.Dataset):
    def __init__(self, size=224, square=False, output_channels=1, max_size_padoutside=None, square_size_padoutside=None, annotated=False, fracturepseudolabled=False, basedir='/home/buehlern/Documents/Masterarbeit/data', projectdir='/home/buehlern/Documents/Masterarbeit', required_cols='auto', cache=False, diskcache_reldir='../../../neocortex-nas/buehlern/Masterarbeit/IDPDatasetBaseCache', diskcache_reldir_autoappend=True, return_df_row=False, return_custom_cols=(), normalization_mode=0.99, bodypartexamined_mappingloc='data/BodyPartExamined_mappings_mergemore.json', bodypartexamined_dropna=False, clean_brightedges=False, clean_rotation=False, merge_scapula_shoulder=False, no_pixelarray_loading=False, total_size=None, ratelimiting=False):
        """
        normalization_mode (float|'max'|None): None means no normalization is applied (the conversion to a float32 tensor nevertheless takes place), float: output is a 0-1-clipped normalization where >= normalization_mode quantile is 1
        """

        super().__init__()
        self.basedir = Path(basedir)
        self.size = size
        self.square = square
        self.output_channels = output_channels
        self.projectdir = Path(projectdir)
        # max_size_padoutside overrides size setting if not None
        self.max_size_padoutside = max_size_padoutside
        self.square_size_padoutside = square_size_padoutside
        self.annotated = annotated
        self.fracturepseudolabled = fracturepseudolabled
        self.cache = cache
        self.diskcache_reldir = diskcache_reldir
        self.return_df_row = return_df_row
        self.return_custom_cols = return_custom_cols
        self.normalization_mode = normalization_mode
        self.clean_brightedges = clean_brightedges
        self.clean_rotation = clean_rotation
        self.no_pixelarray_loading = no_pixelarray_loading
        self.total_size = total_size
        self.ratelimiting = ratelimiting

        if self.clean_brightedges and (self.max_size_padoutside is not None):
            raise ValueError('clean_rotation requires max_size_padoutside=None')
        if self.clean_rotation and (not self.clean_brightedges):
            raise ValueError('clean_rotation requires clean_brightedges=True')
        if self.clean_brightedges and self.size != 224:
            print(
                f'[WARN] clean_brightedges with non-default image size {self.size} instead of {224} may not yield good results')

        if self.fracturepseudolabled and (not self.annotated):
            raise ValueError('fracturepseudolabled requires annotated=True')

        if required_cols == 'auto':
            if not self.annotated:
                required_cols = ('path', 'bodypart', 'patientid', 'examinationid', 'findingspath', 'findings',
                                 'dcm_SOPInstanceUID', 'dcm_PatientID', 'dcm_BodyPartExamined', 'pixelarr_shape')
            elif not self.fracturepseudolabled:
                required_cols = ('path', 'bodypart', 'fracture', 'foreignmaterial', 'bodypart_original', 'annotated',
                                 'patientid', 'examinationid', 'findingspath', 'findings', 'dcm_SOPInstanceUID', 'dcm_PatientID', 'dcm_BodyPartExamined', 'pixelarr_shape')
            else:
                required_cols = ('path', 'bodypart', 'fracture', 'foreignmaterial', 'bodypart_original', 'annotated',
                                 'patientid', 'examinationid', 'findingspath', 'findings', 'dcm_SOPInstanceUID', 'dcm_PatientID', 'dcm_BodyPartExamined', 'pixelarr_shape',
                                 'fracturenum', 'fracture_pseudolabel', 'fracture_bestlabel', 'fracture_bestlabeltext')

        print('initializing IDPDatasetBase ...')
        if self.diskcache_reldir is not None:
            self.diskcache_reldir = Path(self.diskcache_reldir)
            if diskcache_reldir_autoappend:
                self.diskcache_reldir = self.diskcache_reldir.with_name(self.diskcache_reldir.name +
                                                                        ('_cleanbrightedges' if self.clean_brightedges else '') +
                                                                        ('_cleanrotation' if self.clean_rotation else '') + 
                                                                        (f'_{self.size}' if self.max_size_padoutside is None and self.size != 224 else ''))
            if not (self.basedir / self.diskcache_reldir).exists():  # if is symlink crashes with just the mkdir(exist_ok=True line)
                (self.basedir / self.diskcache_reldir).mkdir(exist_ok=True)
            self._getpixelarray_load_funcstrsha256 = hashlib.sha256(
                inspect.getsource(self._getpixelarray_load).encode('utf8')).hexdigest()

        def getdf(mayuseslow=True):
            dataset_thin_loc = self.projectdir / Path(
                f'data/cache-full/dataset_thin{"_annotated" if self.annotated else ""}{"_fracturepseudolabled" if self.fracturepseudolabled else ""}.pt')
            dataset_full_loc = self.projectdir / Path(
                f'data/clean_df{"_annotated" if self.annotated else ""}{"_fracturepseudolabled" if self.fracturepseudolabled else ""}.pkl')

            def getdf_slow():
                assert mayuseslow
                print(f'reading {dataset_full_loc} file' +
                      (f' and re-generating {dataset_thin_loc} ' if required_cols is not None else '') + ' ...')
                #print(dataset_full_loc.resolve())
                df_full = pd.read_pickle(dataset_full_loc)
                if required_cols is None:
                    return df_full

                df = df_full.copy()
                df = df[list(required_cols)]
                torch.save((required_cols, df), dataset_thin_loc)
                return getdf(mayuseslow=False)

            if required_cols is None or not dataset_thin_loc.exists():
                return getdf_slow()

            read_requiredcols, df = torch.load(dataset_thin_loc)
            if read_requiredcols != required_cols:
                return getdf_slow()

            if mayuseslow:
                print(f'fast initialization from {dataset_thin_loc}')
            return df

        df_full = getdf()
        df = df_full.copy()


        # opt. limit total dataset size
        if self.total_size is not None:
            size_per_bodypart = int(self.total_size / len(df['bodypart'].unique()))
            print("Limiting dataset total size to", self.total_size)
            print("Size for each bodypart:", size_per_bodypart)
            df = df.groupby('bodypart', group_keys=False).apply(lambda x: x.sample(min(len(x), size_per_bodypart)))
            print("New length of df:", len(df))

        if bodypartexamined_mappingloc is not None:
            print("PATH", (projectdir / Path(bodypartexamined_mappingloc)).resolve())
            bodypartexamined_mapping = json.loads((projectdir / Path(bodypartexamined_mappingloc)).read_text())
            df['dcm_BodyPartExamined_str'] = [(seq if not seq != seq else '').lower()
                                              for seq in df['dcm_BodyPartExamined']]

            df['mapped_BodyPartExamined'] = [bodypartexamined_mapping[BodyPartExamined_str.replace(' ', '').lower()]
                                             for BodyPartExamined_str in df['dcm_BodyPartExamined_str']]

            if bodypartexamined_dropna:
                df = df.dropna(subset=['mapped_BodyPartExamined'])

        df_labelcomparison_loc = self.projectdir / Path('data/cache-full/df_labelcomparison.pkl')
        if df_labelcomparison_loc.exists():
            df_labelcomparison = pd.read_pickle()

            df_full['dcm_BodyPartExamined_str'] = [(seq if not seq != seq else '').lower()
                                                   for seq in df_full['dcm_BodyPartExamined']]

            filtermask = (df_labelcomparison.fillna(
                0) - df_labelcomparison.fillna(0).astype(int)) > 0.1
            filter_rowcol_tuples = list(
                zip(*np.nonzero(filtermask.to_numpy())))

            for row, col in filter_rowcol_tuples:
                dcmbodypart = list(df_labelcomparison.index)[row]
                bodypart = list(df_labelcomparison.columns)[col][1]

                df = df[~((df['dcm_BodyPartExamined_str'] == dcmbodypart)
                          & (df['bodypart'] == bodypart))]
            print(f'{len(df_full)-len(df)=} items excluded by df_labelcomparison.pkl')
        else:
            print(
                f'{df_labelcomparison_loc} does not exit --> no items excluded by it')

        df['relpathstr'] = helpers.ensureunique(
            [os.path.relpath(path, basedir) for path in df['path']])
        df = df.sort_values('relpathstr')
        df = df.reset_index(drop=True)
        self.bodypart_to_idx = {
            bodypart: i for i, bodypart in enumerate(sorted(set(df['bodypart'])))
        }
        self.idx_to_bodypart = {
            i: bodypart for i, bodypart in enumerate(sorted(set(df['bodypart'])))
        }

        self.fracturepseudolables = ['0_no', '1_yes', '2_votesfew', '3_contradiction', '4_unsure']
        self.fracture_to_idx = {
            bodypart: i for i, bodypart in enumerate(self.fracturepseudolables[:2])
        }

        if merge_scapula_shoulder:
            df['bodypart'] = df['bodypart'].replace('scapula', 'schulter')
            print('replaced all scapula bodyparts with shoulder')

        self.df = df.copy()

        print(self, 'initialized')

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        return f'IDPDatasetBase(len={len(self.df)})'

    def __str__(self) -> str:
        return repr(self)

    @functools.lru_cache
    def _getitem_innercached(self, index):
        return self._getitem_inner(index)

    def _getpixelarray_load(self, curitem_series):
        try:
            dcm = pydicom.dcmread(curitem_series['path'])
        except (AttributeError, OSError):
            null_pixel_array = np.ones((1, 1)) * np.nan
            return torch.tensor(null_pixel_array, dtype=torch.float32)[None]

        pixel_array = dcm.pixel_array
        # normalize by max
        if self.normalization_mode == None:
            pass
        elif self.normalization_mode == 'max':
            pixel_array /= pixel_array.max()
        elif isinstance(self.normalization_mode, float) or isinstance(self.normalization_mode, int):
                pixel_array = pixel_array.astype(float)
                pixel_array /= np.quantile(pixel_array, self.normalization_mode)
                pixel_array = np.clip(pixel_array, 0, 1)

        # add batch dim
        pixel_array = torch.tensor(pixel_array, dtype=torch.float32)[None]

        if self.max_size_padoutside is not None:
            #pixel_array = TF.resize(pixel_array, size=self.max_size_padoutside - 1, max_size=self.max_size_padoutside)
            max_size = self.max_size_padoutside # max(pixel_array.shape)
            missing_cols = max_size - pixel_array.shape[-1]
            pad_left = missing_cols // 2
            pad_right = sum(divmod(missing_cols, 2))

            missing_rows = max_size - pixel_array.shape[-2]
            pad_top = missing_rows // 2
            pad_bottom = sum(divmod(missing_rows, 2))

            pixel_array = F.pad(pixel_array, [0, pad_left+pad_right, 0, pad_top+pad_bottom])
        else:
            # typical case for augmented trainings, allow for varying image sizes from here
            if self.size is not None:
                if self.square:
                    pixel_array = TF.resize(pixel_array, (self.size, self.size))
                else:
                    pixel_array = TF.resize(pixel_array, self.size)

            if self.clean_brightedges:
                def _clean(pixel_array):
                    pixel_array = np.array(pixel_array)
                    img_8bit = cv2.normalize(pixel_array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                    _, thresh = cv2.threshold(img_8bit, 0, 1, cv2.THRESH_BINARY)

                    contours, hierarchy = cv2.findContours(
                        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                    page = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # has to be largest area match

                    # draw contours on the original image
                    contours_only = np.zeros_like(pixel_array)
                    cv2.drawContours(image=contours_only, contours=page,
                                     contourIdx=-1, color=1., thickness=12, lineType=cv2.LINE_AA)

                    if len(page) > 0:
                        # Loop over the contours.
                        c = page[0]
                        # Approximate the contour.
                        epsilon = 0.02 * cv2.arcLength(c, True)
                        corners = cv2.approxPolyDP(c, epsilon, True)
                        # If our approximated contour has four points

                    if len(page) == 0 or len(corners) != 4:
                        # print('[WARN] failed to find 4 corners; fallback to outer border cropping')
                        corners = [
                            [0, 0],
                            [0, pixel_array.shape[0] - 1],
                            [pixel_array.shape[1] - 1, pixel_array.shape[0] - 1],
                            [pixel_array.shape[1] - 1, 0],
                        ]
                        corners = np.array(corners)[:, None, :]
                        contours_only = np.zeros_like(pixel_array)
                        cv2.drawContours(image=contours_only, contours=[corners],
                                         contourIdx=-1, color=1., thickness=12, lineType=cv2.LINE_AA)

                        # raise NotImplementedError('TODO')

                    # import time
                    # starttime = time.time()
                    img_masked = pixel_array * (contours_only == 0) + pixel_array * 0.0 * (contours_only != 0)
                    img_blurred = cv2.GaussianBlur(img_masked, (35, 35), 5.)
                    contours_only = cv2.GaussianBlur(contours_only, (15, 15), 0)
                    # endtime = time.time()
                    # print(endtime - starttime)
                    img_combined = pixel_array * (1 - contours_only) + img_blurred * (contours_only)

                    # plt.imshow(img_combined, cmap='bone')
                    # plt.show()
                    if not self.clean_rotation:
                        return img_combined

                    corners = sorted(np.concatenate(corners).tolist())
                    corners = order_points(corners)
                    corners = np.array(corners)
                    # Calculate the angle of rotation needed to align the image with the x-axis
                    angle_calc = np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]) * 180 / np.pi

                    # Calculate the angle of rotation needed to align the image with the x-axis
                    angle_x = np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]) * 180 / np.pi

                    # Calculate the angle of rotation needed to align the image with the y-axis
                    angle_y = np.arctan2(corners[2, 0] - corners[1, 0], corners[2, 1] - corners[1, 1]) * 180 / np.pi
                    angle_y *= -1

                    angle_calc = sorted([angle_x, angle_y], key=abs)[0]

                    angle = angle_calc if -45 < angle_calc < 45 else 0
                    rotation_matrix = cv2.getRotationMatrix2D((img_8bit.shape[1] / 2, img_8bit.shape[0] / 2), angle, 1)
                    img_combined_rotated = cv2.warpAffine(img_combined, rotation_matrix, img_combined.shape[::-1])

                    return img_combined_rotated

                pixel_array = _clean(pixel_array[0, :, :])
                # re-add batch dim
                pixel_array = torch.tensor(pixel_array, dtype=torch.float32)[None]
        return pixel_array

    def _getpixelarray(self, index, curitem_series):
        if self.diskcache_reldir is not None:
            s = os.stat(curitem_series['path'])
            cur_equalconfig_dict = {
                'index': index,
                # 'curitem_series': curitem_series,
                'path': curitem_series['path'],
                'path_stat': {k: getattr(s, k) for k in dir(s) if k.startswith('st_') and not k.startswith('st_atime')},
                'self.normalization_mode': self.normalization_mode,
                'self.max_size_padoutside': self.max_size_padoutside,
                'self.size': self.size,
                'self.clean_brightedges': self.clean_brightedges,
                'self.clean_rotation': self.clean_rotation,
                'self._getpixelarray_load_funcstrsha256': self._getpixelarray_load_funcstrsha256,
            }

            # try to read from cache first
            cacheloc = self.basedir / self.diskcache_reldir / f'{index}.pt'
            if cacheloc.exists():
                for _ in range(4):
                    try:
                        equalconfig_dict, pixel_array = torch.load(cacheloc)
                        if cur_equalconfig_dict == equalconfig_dict:
                            return pixel_array
                        break
                    except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
                        print('[warn] Error encountered while reading cached image file', index, e)
                        if self.ratelimiting:
                            time.sleep(5)
                        pass
                helpers.print_ratelimited('[warn] unable to read or outdated cached image file')

        pixel_array = self._getpixelarray_load(curitem_series)

        if self.diskcache_reldir is not None:
            try:
                torch.save((cur_equalconfig_dict, pixel_array), cacheloc)
            except RuntimeError as e:
                print('[warn] RuntimeError encountered while writing cached image file', index, e)
        
        # TODO Pad pixel_array to batch size?
        # something like batch_shape = self.batch_shapes[index]

        return pixel_array

    def _getitem_inner(self, index):
        curitem_series = self.df.loc[index]
        pixel_array = self._getpixelarray(index, curitem_series) if not self.no_pixelarray_loading else None
        res = dict(pixel_array=pixel_array, bodypart_idx=self.bodypart_to_idx[curitem_series['bodypart']])
        if self.annotated:
            res['fracture'] = curitem_series['fracture']
        if self.fracturepseudolabled:
            res['fracture_bestlabel'] = curitem_series['fracture_bestlabel']
        if self.return_df_row:
            res['row'] = curitem_series

        for customcol in self.return_custom_cols:
            res[customcol] = curitem_series[customcol]
        return res

    def __getitem__(self, index):
        if self.cache:
            item = self._getitem_innercached(index)
            return item['pixel_array'].expand(self.output_channels, -1, -1), item['bodypart_idx']
        else:
            item = self._getitem_inner(index)
            return item['pixel_array'].expand(self.output_channels, -1, -1), item['bodypart_idx']


def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    # https://learnopencv.com/automatic-document-scanner-using-opencv/
    # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18

    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()

class IDPDataset(torch.utils.data.Dataset):
    def __init__(self, dsbase, mode, stratification_target='bodypart', seed=42, total_size=None, val_size=0.2, test_size=0.2, extra_filter=None):
        super().__init__()
        self.dsbase = dsbase
        self.mode = mode
        self.total_size = total_size
        print(f'\ninitializing IDPDataset(mode={self.mode}) ...')

        if not mode in ['train', 'val', 'test', 'train+val', 'train+val+test']:
            raise ValueError('invalid IDPDataset mode')
        modeset = set(mode.split('+'))
        self.modeset = modeset

        stratification_target_frequencies = dsbase.df[stratification_target].value_counts()
        # if multiple stratification target values for the same patient, use the rarest one for stratification
        # this computation is always performed globally

        self.df = dsbase.df.copy()
        self.df.reset_index(names='dsbase_index', inplace=True)

        stratification_target_frequencies = dsbase.df[stratification_target].value_counts()
        # if multiple stratification target values for the same patient, use the rarest one for stratification
        # this computation is always performed globally

        split_test_loc = self.dsbase.projectdir / Path(f'data/split_test_straton_{stratification_target}_{self.total_size}.csv')
        if not split_test_loc.exists():
            res = input(
                f'WARN: NO TRAINVAL TEST SPLIT FOUND AT {split_test_loc}, type YESGENERATE[enter] to generate one: ')
            if res.strip() != 'YESGENERATE':
                self.df = None
                exit(1)

            print('WARN: GENERATING NEW TRAINVAL TEST SPLIT')
            patientid_to_strattarget = {patientid: sorted(set(subdf[stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in self.df.groupby('dcm_PatientID')}

            _, test = train_test_split(list(patientid_to_strattarget.keys()),
                                       test_size=test_size, stratify=list(patientid_to_strattarget.values()), random_state=0)
            test_patientids = pd.DataFrame(test)
            test_patientids.to_csv(split_test_loc)
        test_patientids = pd.read_csv(split_test_loc, index_col=0)

        patientid_index_df = self.df.set_index('dcm_PatientID')
        #print("patientid_index_df.index", len(patientid_index_df.index))
        #print("test_patientids['0']", len(test_patientids['0']))
        assert set(patientid_index_df.index).issuperset(test_patientids['0'])
        test_idxs = patientid_index_df.loc[test_patientids['0']]['dsbase_index']
        if 'test' in modeset:
            print('WARN: including test data')
            if modeset == {'test'}:
                # remove trainval
                self.df = self.df.loc[test_idxs]
                assert len(set(self.df['dcm_PatientID']) - set(test_patientids['0'])) == 0
            assert len(set(self.df['dcm_PatientID']) & set(test_patientids['0'])) == len(test_patientids)
        else:
            for idx in test_idxs:
                if idx not in self.df.index:
                    print(idx, "not in index!")
            self.df = self.df.drop(test_idxs)
            assert len(set(self.df['dcm_PatientID']) & set(test_patientids['0'])) == 0

        patientid_index_df = self.df.set_index('dcm_PatientID')
        if ('train' in modeset or 'val' in modeset) and not ('train' in modeset and 'val' in modeset):
            patientid_to_strattarget = {patientid: sorted(set(subdf[stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in self.df.groupby('dcm_PatientID')}
                
            train, val = train_test_split(list(patientid_to_strattarget.keys()),
                                          test_size=val_size, stratify=list(patientid_to_strattarget.values()), random_state=seed)
            train_patientids = pd.DataFrame(train).rename(columns={0: '0'})
            val_patientids = pd.DataFrame(val).rename(columns={0: '0'})
            val_idxs = patientid_index_df.loc[val_patientids['0']]['dsbase_index']
            if 'val' in modeset:
                # since not both, only keep the val ones
                self.df = self.df.loc[val_idxs]
                assert len(set(self.df['dcm_PatientID']) & set(val_patientids['0'])) == len(val_patientids)
                assert len(set(self.df['dcm_PatientID']) - set(val_patientids['0'])) == 0
            else:
                self.df = self.df.drop(val_idxs)
                assert len(set(self.df['dcm_PatientID']) & set(train_patientids['0'])) == len(train_patientids)
                assert len(set(self.df['dcm_PatientID']) & set(val_patientids['0'])) == 0

        if extra_filter is not None:
            self.df = extra_filter(self.df)
        self.df = self.df.copy()
        print(self, 'initialized')

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        return f'IDPDataset(mode={self.mode}, len={len(self.df)})'

    def __str__(self) -> str:
        return repr(self)

    def __getitem__(self, index):
        return self.dsbase[self.df.iloc[index]['dsbase_index']]

    def getrow(self, index):
        return self.df.iloc[index]