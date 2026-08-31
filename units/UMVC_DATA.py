import argparse

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


class Unpaired_MultiviewDataset(Dataset):
    """
    Multi-view dataset with deliberately shuffled cross-view correspondences.

    The returned format is kept compatible with the original incomplete-view
    dataset: __getitem__ returns (data, mask, index). Here the mask is all ones,
    because no view is removed; only the sample correspondence across views is
    disturbed.
    """

    def __init__(self, data_list, mask_matrix, labels, num_views, pair_indices=None, unpaired_indices=None):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, num_views, axis=1)
        self.pair_indices = pair_indices
        self.unpaired_indices = unpaired_indices

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32) for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False) for v in range(self.num_views)]
        return data, mask, index


def _derange_indices(indices, rng):
    """
    Generate a derangement on the given indices as much as possible.

    For unpair_rate > 0, the selected samples are shuffled within themselves.
    When there are at least two selected samples, this function guarantees that
    no selected sample keeps its original counterpart in the shuffled view.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= 1:
        return indices.copy()

    for _ in range(100):
        candidate = indices[rng.permutation(len(indices))]
        if np.all(candidate != indices):
            return candidate

    return np.roll(indices, 1)


def build_unpaired_views(data_list, unpair_rate, seed=1, anchor_view=0):
    """
    Shuffle cross-view correspondence for a given proportion of samples.

    Example for two views:
    originally, row 1 is paired as (view1[1], view2[1]) and row 2 is paired as
    (view1[2], view2[2]); after unpairing, they can become
    (view1[1], view2[2]) and (view1[2], view2[1]).

    Parameters
    ----------
    data_list : list[np.ndarray]
        Multi-view features. Each item has shape [N, D_v]. The order of samples
        is assumed to be originally aligned across views.
    unpair_rate : float
        Proportion of samples whose cross-view correspondence should be broken.
        unpair_rate=0.2 means 20% of samples are mismatched across views.
    seed : int
        Random seed for reproducible unpairing.
    anchor_view : int
        The view used as the anchor. Its sample order is not changed. Labels are
        also aligned to this anchor view.

    Returns
    -------
    unpaired_data_list : list[np.ndarray]
        Data list after cross-view shuffling. No feature values are masked.
    pair_indices : np.ndarray
        Shape [N, V]. pair_indices[i, v] records which original sample index is
        used by view v at output row i. For the anchor view, pair_indices[:, v]=arange(N).
    unpaired_indices : np.ndarray
        Indices selected to be unpaired.
    """
    assert 0.0 <= unpair_rate <= 1.0, 'unpair_rate should be in [0, 1].'
    num_views = len(data_list)
    assert num_views >= 2, 'The number of views should be at least 2.'

    data_size = data_list[0].shape[0]
    for v, view_data in enumerate(data_list):
        assert view_data.shape[0] == data_size, f'View {v} has inconsistent sample size.'

    assert 0 <= anchor_view < num_views, 'anchor_view should be in [0, num_views).'

    rng = np.random.default_rng(seed)
    unpair_sample_num = int(np.floor(data_size * unpair_rate))

    all_indices = np.arange(data_size, dtype=np.int64)
    if unpair_sample_num > 0:
        unpaired_indices = rng.choice(all_indices, size=unpair_sample_num, replace=False)
    else:
        unpaired_indices = np.array([], dtype=np.int64)

    pair_indices = np.tile(all_indices.reshape(-1, 1), (1, num_views))

    # Keep the anchor view unchanged. Shuffle each non-anchor view independently
    # within the selected unpaired subset. This preserves every view's sample set
    # and only destroys cross-view one-to-one correspondence.
    for v in range(num_views):
        if v == anchor_view:
            continue
        shuffled_indices = _derange_indices(unpaired_indices, rng)
        pair_indices[unpaired_indices, v] = shuffled_indices

    unpaired_data_list = [data_list[v][pair_indices[:, v]] for v in range(num_views)]

    actual_unpaired = np.zeros(data_size, dtype=bool)
    for v in range(num_views):
        if v == anchor_view:
            continue
        actual_unpaired |= (pair_indices[:, v] != all_indices)

    print(f'Unpair rate: {unpair_rate:.4f}')
    print(f'Requested unpaired samples: {unpair_sample_num}/{data_size}')
    print(f'Actual unpaired samples: {int(actual_unpaired.sum())}/{data_size}')

    return unpaired_data_list, pair_indices, unpaired_indices


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


def build_dataset_unpair(args):
    """
    Build an unpaired multi-view dataset.

    Difference from the original missing-view version:
    1. No missing mask is generated.
    2. No feature matrix is multiplied by a 0/1 mask.
    3. A proportion of samples controlled by args.unpair_rate is selected, and
       their non-anchor views are shuffled within the selected subset.

    The returned mask is all ones for compatibility with training code that
    expects a mask input.
    """
    data_list, labels = load_ml_data(args)
    args.unpair_rate = args.missing_rate
    seed = getattr(args, 'seed', 1)
    anchor_view = getattr(args, 'anchor_view', 0)
    data_list, pair_indices, unpaired_indices = build_unpaired_views(
        data_list=data_list,
        unpair_rate=args.unpair_rate,
        seed=seed,
        anchor_view=anchor_view,
    )

    mask = np.ones((args.data_size, args.num_views), dtype=np.float32)

    unpaired_multiview_dataset = Unpaired_MultiviewDataset(
        data_list=data_list,
        mask_matrix=mask,
        labels=labels,
        num_views=args.num_views,
        pair_indices=pair_indices,
        unpaired_indices=unpaired_indices,
    )

    return unpaired_multiview_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Build unpaired multi-view dataset.')
    parser.add_argument('--dataset', type=str, default='BDGP', choices=list(data_info.keys()))
    parser.add_argument('--unpair_rate', type=float, default=0.2,
                        help='Proportion of samples with shuffled cross-view correspondence.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for data loading and unpairing.')
    parser.add_argument('--anchor_view', type=int, default=0,
                        help='Anchor view index. This view and labels remain unchanged.')
    return parser.parse_args()





if __name__ == '__main__':
    args = parse_args()
    # dataset = build_dataset(args)
    #
    # print(dataset)
    # print('Dataset length:', len(dataset))
    # print('Pair index matrix shape:', dataset.pair_indices.shape)
    # print('First 10 pair indices:')
    # print(dataset.pair_indices[:10])
