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
    Caltech5V = {1: 'Caltech5V', 'N': 1400, 'K': 7, 'V': 5, 'n_input': [40, 254, 1984, 512, 928]},
)


class Noisy_MultiviewDataset(Dataset):

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

