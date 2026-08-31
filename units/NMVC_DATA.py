import argparse
from units.config import *
import h5py
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
from units.IMVC_DATA import getData

data_info = dict(
    Caltech={1: 'Caltech', 'N': 2386, 'K': 20, 'V': 6, 'n_input': [48, 40, 254, 1984, 512, 928]},
    Scene_15={1: 'Scene_15', 'N': 4485, 'K': 15, 'V': 3, 'n_input': [20, 59, 40]},
    LandUse_21={1: 'LandUse_21', 'N': 2100, 'K': 21, 'V': 3, 'n_input': [20, 59, 40]},
    HW={1: 'HW', 'N': 2000, 'K': 10, 'V': 6, 'n_input': [216, 76, 64, 6, 240, 47]},
    Wiki_fea={1: 'Wiki_fea', 'N': 2866, 'K': 10, 'V': 2, 'n_input': [128, 10]},
    CUB={1: 'CUB', 'N': 600, 'K': 10, 'V': 2, 'n_input': [1024, 300]},
    CCV={1: 'CCV', 'N': 6673, 'K': 5, 'V': 3, 'n_input': [20, 20, 20]},
    NUSWIDE={1: 'NUSWIDE', 'N': 5000, 'K': 5, 'V': 5, 'n_input': [65, 226, 145, 74, 129]},
    PIE_face_10={1: 'PIE_face_10', 'N': 680, 'K': 10, 'V': 3, 'n_input': [484, 256, 279]},
    BBCSport={1: 'BBCSport', 'N': 544, 'K': 5, 'V': 2, 'n_input': [3183, 3203]},
    BDGP={1: 'BDGP', 'N': 2500, 'K': 5, 'V': 2, 'n_input': [1750, 79]},
    NGs={1: 'NGs', 'N': 500, 'K': 5, 'V': 3, 'n_input': [2000, 2000, 2000]},
    Hdigit={1: 'Hdigit', 'N': 10000, 'K': 10, 'V': 2, 'n_input': [784, 256]},
    cora={1: 'cora', 'N': 2708, 'K': 7, 'V': 2, 'n_input': [2708, 1433]},
    cifar10={1: 'cifar10', 'N': 50000, 'K': 10, 'V': 3, 'n_input': [512, 2048, 1024]},
    stl10_fea={1: 'stl10_fea', 'N': 13000, 'K': 10, 'V': 3, 'n_input': [1024, 512, 2048]},
    Reuters_1200={1: 'Reuters', 'N': 1200, 'K': 6, 'V': 5, 'n_input': [2000, 2000, 2000, 2000, 2000]},
    UCI_Digits={1: 'UCI_Digits', 'N': 2000, 'K': 10, 'V': 3, 'n_input': [240, 76, 216, 47, 64, 6]},
    NUSWIDE_deep={1: 'NUSWIDE_deep', 'N': 9000, 'K': 6, 'V': 2, 'n_input': [4096, 300]},
    Caltech101_7={1: 'Caltech101_7', 'N': 1474, 'K': 10, 'V': 6, 'n_input': [48, 40, 254, 1984, 512, 928]},
    Movies={1: 'Movies', 'N': 617, 'K': 17, 'V': 2, 'n_input': [1878, 1398]},
    DHA={1: 'DHA', 'N': 483, 'K': 23, 'V': 2, 'n_input': [110, 6144]},
    ALOI={1: 'ALOI', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125]},
    Caltech5V = {1: 'Caltech5V', 'N': 1400, 'K': 7, 'V': 5, 'n_input': [40, 254, 1984, 512, 928]},
    Reuters_small = {1: 'Reuters_small', 'N': 1200, 'K': 6, 'V': 5, 'n_input': [2000, 2000, 2000, 2000, 2000]},
    NUS_WIDE = {1: 'NUS_WIDE', 'N': 2000, 'K': 31, 'V': 5, 'n_input': [65, 226, 145, 74, 129]},
    MSRC_v1 = {1: 'MSRC_v1', 'N': 210, 'K': 7, 'V': 5, 'n_input': [24, 576, 512, 256, 254]},
    flower17 = {1: 'flower17', 'N': 1360, 'K': 17, 'V': 7, 'n_input': [1360, 1360, 1360, 1360, 1360, 1360, 1360]},
)


class Noisy_MultiviewDataset(Dataset):
    """
    Multi-view dataset with partially noisy views.

    The returned format is compatible with the original incomplete-view/unpaired
    dataset: __getitem__ returns (data, mask, index). Here the mask is all ones,
    because no view is removed; only some existing views are corrupted by noise.

    Extra attributes
    ----------------
    clean_mask : np.ndarray, shape [N, V]
        clean_mask[i, v] = 1 means sample i in view v is clean;
        clean_mask[i, v] = 0 means this view is noisy.
    noisy_mask : np.ndarray, shape [N, V]
        noisy_mask[i, v] = 1 means sample i in view v is noisy.
    noisy_indices : np.ndarray
        Sample indices selected for noise injection.
    """

    def __init__(self, data_list, mask_matrix, labels, num_views,
                 noisy_mask=None, noisy_indices=None):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, num_views, axis=1)
        self.noisy_mask = noisy_mask
        self.noisy_indices = noisy_indices
        if noisy_mask is None:
            self.clean_mask = np.ones_like(mask_matrix, dtype=np.float32)
        else:
            self.clean_mask = (1.0 - noisy_mask.astype(np.float32))

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32)
                for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False)
                for v in range(self.num_views)]
        return data, mask, index


def add_noise_to_view(view_data, noise_type='gaussian', noise_std=0.1,
                      salt_pepper_ratio=0.1, rng=None):
    """
    Add one specified type of noise to a single view feature matrix.

    Parameters
    ----------
    view_data : np.ndarray
        Feature matrix to be corrupted, shape [M, D].
    noise_type : str
        Noise type. Supported values:
        - 'gaussian': x + N(0, noise_std^2)
        - 'uniform': x + U(-noise_std, noise_std)
        - 'salt_pepper': randomly set entries to feature min/max values
        - 'dropout': randomly set entries to 0
    noise_std : float
        Noise strength for gaussian/uniform noise.
    salt_pepper_ratio : float
        Corrupted feature-entry ratio for salt_pepper/dropout noise.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    noisy_view_data : np.ndarray
        Corrupted feature matrix with the same shape as input.
    """
    if rng is None:
        rng = np.random.default_rng(1)

    x = view_data.astype(np.float32, copy=True)
    noise_type = str(noise_type).lower()

    if x.size == 0:
        return x

    if noise_type == 'gaussian':
        noise = rng.normal(loc=0.0, scale=noise_std, size=x.shape).astype(np.float32)
        x = x + noise

    elif noise_type == 'uniform':
        noise = rng.uniform(low=-noise_std, high=noise_std, size=x.shape).astype(np.float32)
        x = x + noise

    elif noise_type == 'salt_pepper':
        ratio = float(np.clip(salt_pepper_ratio, 0.0, 1.0))
        entry_mask = rng.random(x.shape) < ratio
        salt_mask = rng.random(x.shape) < 0.5
        min_val = np.min(x, axis=0, keepdims=True)
        max_val = np.max(x, axis=0, keepdims=True)
        x[entry_mask & salt_mask] = np.broadcast_to(max_val, x.shape)[entry_mask & salt_mask]
        x[entry_mask & (~salt_mask)] = np.broadcast_to(min_val, x.shape)[entry_mask & (~salt_mask)]

    elif noise_type == 'dropout':
        ratio = float(np.clip(salt_pepper_ratio, 0.0, 1.0))
        entry_mask = rng.random(x.shape) < ratio
        x[entry_mask] = 0.0

    else:
        raise ValueError(
            f"Unsupported noise_type: {noise_type}. "
            "Choose from ['gaussian', 'uniform', 'salt_pepper', 'dropout']."
        )

    return x.astype(np.float32)


def _choose_noisy_views_for_samples(data_size, num_views, noisy_indices, rng,
                                    min_noisy_views=1, max_noisy_views=None):
    """
    Select which view(s) should be noisy for each selected sample.

    Important constraint:
    For every noisy sample, at least one view remains clean. Therefore, the
    number of noisy views per selected sample is at most V - 1. For two-view
    data, each selected sample has exactly one noisy view and one clean view.
    """
    assert num_views >= 2, 'The number of views should be at least 2.'

    if max_noisy_views is None:
        max_noisy_views = num_views - 1

    min_noisy_views = int(max(1, min_noisy_views))
    max_noisy_views = int(min(max_noisy_views, num_views - 1))
    assert min_noisy_views <= max_noisy_views, 'min_noisy_views should be <= max_noisy_views.'

    noisy_mask = np.zeros((data_size, num_views), dtype=np.float32)
    for idx in noisy_indices:
        noisy_view_num = rng.integers(min_noisy_views, max_noisy_views + 1)
        noisy_views = rng.choice(np.arange(num_views), size=noisy_view_num, replace=False)
        noisy_mask[idx, noisy_views] = 1.0

    return noisy_mask


def build_noisy_views(data_list, noisy_rate, seed=1, noise_type='gaussian',
                      noise_std=0.1, salt_pepper_ratio=0.1,
                      min_noisy_views=1, max_noisy_views=None):
    """
    Add noise to a given proportion of multi-view samples.

    Difference from the unpaired version:
    1. Sample correspondence across views is not shuffled.
    2. No view is removed, so the training mask remains all ones.
    3. For each selected sample, only part of its views are corrupted, while at
       least one view remains clean. Example for two views: (v1 clean, v2 noisy)
       or (v1 noisy, v2 clean).

    Parameters
    ----------
    data_list : list[np.ndarray]
        Multi-view features. Each item has shape [N, D_v].
    noisy_rate : float
        Proportion of samples that contain noisy view(s). In this file,
        args.missing_rate is directly used as noisy_rate.
    seed : int
        Random seed for reproducible noise injection.
    noise_type : str
        Noise type passed to add_noise_to_view().
    noise_std : float
        Strength for gaussian/uniform noise.
    salt_pepper_ratio : float
        Entry corruption ratio for salt_pepper/dropout noise.
    min_noisy_views : int
        Minimum number of noisy views for each selected sample.
    max_noisy_views : int or None
        Maximum number of noisy views for each selected sample. It will be
        clipped to V - 1 to guarantee at least one clean view.

    Returns
    -------
    noisy_data_list : list[np.ndarray]
        Data list after partial noise injection.
    noisy_mask : np.ndarray
        Shape [N, V]. noisy_mask[i, v] = 1 indicates view v of sample i is noisy.
    noisy_indices : np.ndarray
        Indices selected to contain at least one noisy view.
    """
    assert 0.0 <= noisy_rate <= 1.0, 'noisy_rate should be in [0, 1].'
    num_views = len(data_list)
    assert num_views >= 2, 'The number of views should be at least 2.'

    data_size = data_list[0].shape[0]
    for v, view_data in enumerate(data_list):
        assert view_data.shape[0] == data_size, f'View {v} has inconsistent sample size.'

    rng = np.random.default_rng(seed)
    noisy_sample_num = int(np.floor(data_size * noisy_rate))

    all_indices = np.arange(data_size, dtype=np.int64)
    if noisy_sample_num > 0:
        noisy_indices = rng.choice(all_indices, size=noisy_sample_num, replace=False)
    else:
        noisy_indices = np.array([], dtype=np.int64)

    noisy_mask = _choose_noisy_views_for_samples(
        data_size=data_size,
        num_views=num_views,
        noisy_indices=noisy_indices,
        rng=rng,
        min_noisy_views=min_noisy_views,
        max_noisy_views=max_noisy_views,
    )

    noisy_data_list = [view.copy().astype(np.float32) for view in data_list]
    for v in range(num_views):
        row_indices = np.where(noisy_mask[:, v] == 1)[0]
        if len(row_indices) == 0:
            continue
        noisy_data_list[v][row_indices] = add_noise_to_view(
            noisy_data_list[v][row_indices],
            noise_type=noise_type,
            noise_std=noise_std,
            salt_pepper_ratio=salt_pepper_ratio,
            rng=rng,
        )

    print(f'Noisy rate: {noisy_rate:.4f}')
    print(f'Requested noisy samples: {noisy_sample_num}/{data_size}')
    print(f'Actual noisy samples: {int((noisy_mask.sum(axis=1) > 0).sum())}/{data_size}')
    print(f'Noise type: {noise_type}')
    print(f'Noisy view entries: {int(noisy_mask.sum())}/{data_size * num_views}')

    return noisy_data_list, noisy_mask, noisy_indices


def load_ml_data(args):
    dataset_para = data_info[args.dataset]
    data, labels = getData(dataset_para)

    args.multiview_dims = [dv.shape[1] for dv in data]
    args.num_views = len(data)
    args.class_num = len(np.unique(labels))
    args.data_size = labels.shape[0]
    args.z_dim = args.class_num

    if np.max(labels) == args.class_num:
        labels = labels - 1

    print(f'Number of views:{len(data)}\nNumber of samples:{len(labels)}\nNumber of class:{len(np.unique(labels))}')
    print(args.multiview_dims)

    return data, labels


def build_dataset_nmvc(args):
    """
    Build a noisy multi-view clustering dataset.

    args.missing_rate is reused as noisy_rate for compatibility with existing
    experiment scripts. The generated data are still complete multi-view data,
    but a proportion of samples contain noise in only part of their views.
    """
    data_list, labels = load_ml_data(args)

    args.noisy_rate = args.missing_rate
    seed = getattr(args, 'seed', 1)
    noise_type = getattr(args, 'noise_type', 'gaussian')
    noise_std = getattr(args, 'noise_std', 0.1)
    salt_pepper_ratio = getattr(args, 'salt_pepper_ratio', 0.1)
    min_noisy_views = getattr(args, 'min_noisy_views', 1)
    max_noisy_views = getattr(args, 'max_noisy_views', None)

    data_list, noisy_mask, noisy_indices = build_noisy_views(
        data_list=data_list,
        noisy_rate=args.noisy_rate,
        seed=seed,
        noise_type=noise_type,
        noise_std=noise_std,
        salt_pepper_ratio=salt_pepper_ratio,
        min_noisy_views=min_noisy_views,
        max_noisy_views=max_noisy_views,
    )

    # All views still exist. The mask is kept all-ones for compatibility with
    # existing training code. Use dataset.noisy_mask or dataset.clean_mask if you
    # need to know which sample-view entries are noisy/clean.
    mask = np.ones((args.data_size, args.num_views), dtype=np.float32)

    noisy_multiview_dataset = Noisy_MultiviewDataset(
        data_list=data_list,
        mask_matrix=mask,
        labels=labels,
        num_views=args.num_views,
        noisy_mask=noisy_mask,
        noisy_indices=noisy_indices,
    )

    return noisy_multiview_dataset


# Optional alias. If your original training script imports build_dataset(args),
# this keeps the call unchanged.
def build_dataset(args):
    return build_dataset_nmvc(args)


# def parse_args():
#     parser = argparse.ArgumentParser(description='Build noisy multi-view dataset.')
#     parser.add_argument('--dataset', type=str, default='BDGP', choices=list(data_info.keys()))
#     parser.add_argument('--missing_rate', type=float, default=0.2,
#                         help='Used as noisy_rate in NMVC_DATA.py.')
#     parser.add_argument('--seed', type=int, default=1,
#                         help='Random seed for data loading and noise injection.')
#     parser.add_argument('--noise_type', type=str, default='gaussian',
#                         choices=['gaussian', 'uniform', 'salt_pepper', 'dropout'],
#                         help='Noise type applied to selected sample-view entries.')
#     parser.add_argument('--noise_std', type=float, default=0.1,
#                         help='Noise strength for gaussian/uniform noise.')
#     parser.add_argument('--salt_pepper_ratio', type=float, default=0.1,
#                         help='Feature-entry corruption ratio for salt_pepper/dropout noise.')
#     parser.add_argument('--min_noisy_views', type=int, default=1,
#                         help='Minimum number of noisy views for each selected sample.')
#     parser.add_argument('--max_noisy_views', type=int, default=None,
#                         help='Maximum number of noisy views for each selected sample; clipped to V-1.')
#     return parser.parse_args()




if __name__ == '__main__':
    args = parse_args()
    dataset = build_dataset_nmvc(args)
    print(dataset)
    print('Dataset length:', len(dataset))
    print('Noisy mask shape:', dataset.noisy_mask.shape)
    print('First 10 noisy-mask rows:')
    print(dataset.noisy_mask[:10])
