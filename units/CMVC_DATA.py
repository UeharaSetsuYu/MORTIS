import argparse

import h5py
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
from units.IMVC_DATA import load_ml_data as load_canonical_multiview_data


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
)




class Comprehensive_MultiviewDataset(Dataset):
    """
    Dataset for compound multi-view degradation settings.

    Supported args.data_model values:
    - "UI"  : Unpaired + Incomplete
    - "UN"  : Unpaired + Noisy
    - "IN"  : Incomplete + Noisy
    - "UIN" : Unpaired + Incomplete + Noisy

    The returned format is compatible with the original loaders:
        __getitem__(index) -> (data, mask, index)

    Extra attributes are kept for analysis/debugging:
        mask_matrix, pair_indices, unpaired_indices,
        noisy_mask, noisy_indices, clean_mask, compound_report
    """
    def __init__(self, data_list, mask_matrix, labels, num_views,
                 pair_indices=None, unpaired_indices=None,
                 noisy_mask=None, noisy_indices=None, compound_report=None):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_matrix = mask_matrix.astype(np.float32)
        self.mask_list = np.split(self.mask_matrix, num_views, axis=1)

        self.pair_indices = pair_indices
        self.unpaired_indices = unpaired_indices
        self.noisy_mask = noisy_mask
        self.noisy_indices = noisy_indices
        if noisy_mask is None:
            self.clean_mask = np.ones_like(self.mask_matrix, dtype=np.float32)
        else:
            self.clean_mask = 1.0 - noisy_mask.astype(np.float32)
        self.compound_report = compound_report or {}

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32)
                for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False)
                for v in range(self.num_views)]
        return data, mask, index


def _derange_indices(indices, rng):
    """Generate a derangement on the given indices as much as possible."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= 1:
        return indices.copy()
    for _ in range(100):
        candidate = indices[rng.permutation(len(indices))]
        if np.all(candidate != indices):
            return candidate
    return np.roll(indices, 1)


def _sample_indices(data_size, rate, rng):
    """Sample floor(N * rate) row indices without replacement."""
    assert 0.0 <= rate <= 1.0, 'rate should be in [0, 1].'
    sample_num = int(np.floor(data_size * rate))
    if sample_num <= 0:
        return np.array([], dtype=np.int64)
    return rng.choice(np.arange(data_size, dtype=np.int64), size=sample_num, replace=False)


def build_unpaired_views(data_list, unpair_rate, seed=1, anchor_view=0,
                         selected_indices=None):
    """
    Shuffle non-anchor views for selected samples.
    pair_indices[i, v] records which original row is used by output row i, view v.
    """
    assert 0.0 <= unpair_rate <= 1.0, 'unpair_rate should be in [0, 1].'
    num_views = len(data_list)
    assert num_views >= 2, 'The number of views should be at least 2.'
    data_size = data_list[0].shape[0]
    for v, view_data in enumerate(data_list):
        assert view_data.shape[0] == data_size, f'View {v} has inconsistent sample size.'
    assert 0 <= anchor_view < num_views, 'anchor_view should be in [0, num_views).'

    rng = np.random.default_rng(seed)
    all_indices = np.arange(data_size, dtype=np.int64)
    if selected_indices is None:
        unpaired_indices = _sample_indices(data_size, unpair_rate, rng)
    else:
        unpaired_indices = np.asarray(selected_indices, dtype=np.int64)

    pair_indices = np.tile(all_indices.reshape(-1, 1), (1, num_views))
    for v in range(num_views):
        if v == anchor_view:
            continue
        shuffled_indices = _derange_indices(unpaired_indices, rng)
        pair_indices[unpaired_indices, v] = shuffled_indices

    unpaired_data_list = [data_list[v][pair_indices[:, v]] for v in range(num_views)]
    return unpaired_data_list, pair_indices, unpaired_indices


def get_mask(num_views, data_size, missing_rate, seed=1, selected_indices=None):
    """
    Generate an incomplete-view mask.
    For each selected incomplete sample, at least one view is observed and at least
    one view is missing, following the original IMVC mask logic.
    """
    assert num_views >= 2
    assert 0.0 <= missing_rate <= 1.0, 'missing_rate should be in [0, 1].'
    rng = np.random.default_rng(seed)
    if selected_indices is None:
        miss_ind = _sample_indices(data_size, missing_rate, rng)
    else:
        miss_ind = np.asarray(selected_indices, dtype=np.int64)

    mask = np.ones([data_size, num_views], dtype=np.float32)
    for idx in miss_ind:
        while True:
            rand_v = rng.random(num_views)
            v_threshold = rng.random(1)
            observed_ind = (rand_v >= v_threshold)
            rand_v[observed_ind] = 1.0
            rand_v[~observed_ind] = 0.0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[idx] = rand_v.astype(np.float32)
    return mask, miss_ind


def add_noise_to_view(view_data, noise_type='gaussian', noise_std=0.1,
                      salt_pepper_ratio=0.1, rng=None):
    """Add gaussian/uniform/salt_pepper/dropout noise to a feature matrix."""
    if rng is None:
        rng = np.random.default_rng(1)
    x = view_data.astype(np.float32, copy=True)
    noise_type = str(noise_type).lower()
    if x.size == 0:
        return x

    if noise_type == 'gaussian':
        x = x + rng.normal(loc=0.0, scale=noise_std, size=x.shape).astype(np.float32)
    elif noise_type == 'uniform':
        x = x + rng.uniform(low=-noise_std, high=noise_std, size=x.shape).astype(np.float32)
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
                                    min_noisy_views=1, max_noisy_views=None,
                                    observed_mask=None):
    """
    Select noisy views for each noisy sample.
    If observed_mask is given, noise is injected only into observed views, so a
    zero-filled missing view will not be corrupted again.
    """
    assert num_views >= 2, 'The number of views should be at least 2.'
    if max_noisy_views is None:
        max_noisy_views = num_views - 1
    min_noisy_views = int(max(1, min_noisy_views))
    max_noisy_views = int(max(1, min(max_noisy_views, num_views - 1)))

    noisy_mask = np.zeros((data_size, num_views), dtype=np.float32)
    for idx in noisy_indices:
        if observed_mask is None:
            candidate_views = np.arange(num_views)
            max_allowed = max_noisy_views
        else:
            candidate_views = np.where(observed_mask[idx] > 0)[0]
            # If only one view is observed, corrupt at most that one. If no view is
            # observed, skip. The incomplete mask generator normally prevents this.
            if len(candidate_views) == 0:
                continue
            max_allowed = min(max_noisy_views, len(candidate_views))

        min_allowed = min(min_noisy_views, max_allowed)
        noisy_view_num = rng.integers(min_allowed, max_allowed + 1)
        noisy_views = rng.choice(candidate_views, size=noisy_view_num, replace=False)
        noisy_mask[idx, noisy_views] = 1.0
    return noisy_mask


def build_noisy_views(data_list, noisy_rate, seed=1, noise_type='gaussian',
                      noise_std=0.1, salt_pepper_ratio=0.1,
                      min_noisy_views=1, max_noisy_views=None,
                      selected_indices=None, observed_mask=None):
    """Inject noise into selected sample-view entries."""
    assert 0.0 <= noisy_rate <= 1.0, 'noisy_rate should be in [0, 1].'
    num_views = len(data_list)
    assert num_views >= 2, 'The number of views should be at least 2.'
    data_size = data_list[0].shape[0]
    for v, view_data in enumerate(data_list):
        assert view_data.shape[0] == data_size, f'View {v} has inconsistent sample size.'

    rng = np.random.default_rng(seed)
    if selected_indices is None:
        noisy_indices = _sample_indices(data_size, noisy_rate, rng)
    else:
        noisy_indices = np.asarray(selected_indices, dtype=np.int64)

    noisy_mask = _choose_noisy_views_for_samples(
        data_size=data_size,
        num_views=num_views,
        noisy_indices=noisy_indices,
        rng=rng,
        min_noisy_views=min_noisy_views,
        max_noisy_views=max_noisy_views,
        observed_mask=observed_mask,
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

    print(f'Number of views:{len(data)}')
    print(f'Number of samples:{len(labels)}')
    print(f'Number of class:{len(np.unique(labels))}')
    print(args.multiview_dims)
    return data, labels


def _build_compound_report(mask, pair_indices, noisy_mask, anchor_view=0):
    data_size, num_views = mask.shape
    all_indices = np.arange(data_size, dtype=np.int64)

    missing_sample_bool = (mask.sum(axis=1) < num_views)
    if pair_indices is None:
        unpaired_sample_bool = np.zeros(data_size, dtype=bool)
    else:
        unpaired_sample_bool = np.zeros(data_size, dtype=bool)
        for v in range(num_views):
            if v == anchor_view:
                continue
            unpaired_sample_bool |= (pair_indices[:, v] != all_indices)

    if noisy_mask is None:
        noisy_sample_bool = np.zeros(data_size, dtype=bool)
    else:
        noisy_sample_bool = (noisy_mask.sum(axis=1) > 0)

    # Extreme UI case: at least one view is missing, and at least two observed
    # views are still available but their source indices are not identical.
    ui_extreme_bool = np.zeros(data_size, dtype=bool)
    if pair_indices is not None:
        for i in range(data_size):
            observed_views = np.where(mask[i] > 0)[0]
            if len(observed_views) >= 2 and mask[i].sum() < num_views:
                src = pair_indices[i, observed_views]
                if len(np.unique(src)) > 1:
                    ui_extreme_bool[i] = True

    return {
        'missing_samples': int(missing_sample_bool.sum()),
        'unpaired_samples': int(unpaired_sample_bool.sum()),
        'noisy_samples': int(noisy_sample_bool.sum()),
        'missing_unpaired_overlap': int((missing_sample_bool & unpaired_sample_bool).sum()),
        'missing_noisy_overlap': int((missing_sample_bool & noisy_sample_bool).sum()),
        'unpaired_noisy_overlap': int((unpaired_sample_bool & noisy_sample_bool).sum()),
        'missing_unpaired_noisy_overlap': int((missing_sample_bool & unpaired_sample_bool & noisy_sample_bool).sum()),
        'ui_extreme_missing_and_observed_misaligned': int(ui_extreme_bool.sum()),
    }


def build_dataset_compound(args):
    """
    Build compound multi-view data according to args.data_model.

    args.data_model:
        UI  = unpaired + incomplete
        UN  = unpaired + noisy
        IN  = incomplete + noisy
        UIN = unpaired + incomplete + noisy

    Ratio rule:
        Every active degradation uses the same ratio args.missing_rate.
        For example, when args.data_model == "UI" and args.missing_rate == 0.5,
        50% samples are selected as incomplete and 50% samples are selected as
        unpaired. The selected subsets are sampled independently by default, so
        overlaps are allowed and reflect truly compound cases.
    """
    data_model = str(getattr(args, 'data_model', 'UI')).upper()
    valid_modes = {'UI', 'UN', 'IN', 'UIN'}
    if data_model not in valid_modes:
        raise ValueError(f'Unsupported args.data_model={data_model}. Choose from {sorted(valid_modes)}.')

    data_list, labels = load_canonical_multiview_data(args)
    rate = float(getattr(args, 'missing_rate', 0.0))
    seed = int(getattr(args, 'seed', 1))
    anchor_view = int(getattr(args, 'anchor_view', 0))

    use_unpair = 'U' in data_model
    use_incomplete = 'I' in data_model
    use_noisy = 'N' in data_model

    # Use separate random streams so each degradation has exactly the same rate
    # while remaining independently sampled.
    idx_rng_unpair = np.random.default_rng(seed + 101)
    idx_rng_incomplete = np.random.default_rng(seed + 202)
    idx_rng_noisy = np.random.default_rng(seed + 303)

    data_size = args.data_size
    num_views = args.num_views

    pair_indices = np.tile(np.arange(data_size).reshape(-1, 1), (1, num_views))
    unpaired_indices = np.array([], dtype=np.int64)
    mask = np.ones((data_size, num_views), dtype=np.float32)
    missing_indices = np.array([], dtype=np.int64)
    noisy_mask = np.zeros((data_size, num_views), dtype=np.float32)
    noisy_indices = np.array([], dtype=np.int64)

    if use_unpair:
        unpaired_indices = _sample_indices(data_size, rate, idx_rng_unpair)
        data_list, pair_indices, unpaired_indices = build_unpaired_views(
            data_list=data_list,
            unpair_rate=rate,
            seed=seed + 11,
            anchor_view=anchor_view,
            selected_indices=unpaired_indices,
        )

    if use_incomplete:
        missing_indices = _sample_indices(data_size, rate, idx_rng_incomplete)
        mask, missing_indices = get_mask(
            num_views=num_views,
            data_size=data_size,
            missing_rate=rate,
            seed=seed + 22,
            selected_indices=missing_indices,
        )
        # In modes containing U, row identity and the unified partition are
        # defined by the anchor view. Keeping it observed prevents a row from
        # being represented only by a shuffled sample belonging to another
        # instance. The hidden permutation is never exposed to training.
        preserve_anchor = bool(getattr(args, 'preserve_unpaired_anchor', True))
        if use_unpair and preserve_anchor:
            for row in missing_indices:
                if mask[row, anchor_view] == 0:
                    observed_views = np.where(mask[row] > 0)[0]
                    mask[row, anchor_view] = 1.0
                    if len(observed_views) > 0:
                        mask[row, observed_views[0]] = 0.0
        data_list = [data_list[v] * mask[:, v:v + 1] for v in range(num_views)]

    if use_noisy:
        noisy_indices = _sample_indices(data_size, rate, idx_rng_noisy)
        noise_type = getattr(args, 'noise_type', 'gaussian')
        noise_std = getattr(args, 'noise_std', 0.1)
        salt_pepper_ratio = getattr(args, 'salt_pepper_ratio', 0.1)
        min_noisy_views = getattr(args, 'min_noisy_views', 1)
        max_noisy_views = getattr(args, 'max_noisy_views', None)
        data_list, noisy_mask, noisy_indices = build_noisy_views(
            data_list=data_list,
            noisy_rate=rate,
            seed=seed + 33,
            noise_type=noise_type,
            noise_std=noise_std,
            salt_pepper_ratio=salt_pepper_ratio,
            min_noisy_views=min_noisy_views,
            max_noisy_views=max_noisy_views,
            selected_indices=noisy_indices,
            observed_mask=mask,
        )

    compound_report = _build_compound_report(
        mask=mask,
        pair_indices=pair_indices if use_unpair else None,
        noisy_mask=noisy_mask if use_noisy else None,
        anchor_view=anchor_view,
    )
    compound_report.update({
        'data_model': data_model,
        'rate': rate,
        'requested_each_degradation_samples': int(np.floor(data_size * rate)),
        'missing_indices': missing_indices,
    })

    print(f'Compound data_model: {data_model}')
    print(f'Compound rate for each active degradation: {rate:.4f}')
    print(f'Requested samples for each active degradation: {int(np.floor(data_size * rate))}/{data_size}')
    print('Compound report:', {k: v for k, v in compound_report.items() if not k.endswith("indices")})

    return Comprehensive_MultiviewDataset(
        data_list=data_list,
        mask_matrix=mask,
        labels=labels,
        num_views=num_views,
        pair_indices=pair_indices if use_unpair else None,
        unpaired_indices=unpaired_indices if use_unpair else None,
        noisy_mask=noisy_mask if use_noisy else None,
        noisy_indices=noisy_indices if use_noisy else None,
        compound_report=compound_report,
    )


# Compatible alias for the original training scripts.
def build_dataset_Multi(args):
    return build_dataset_compound(args)


def parse_args():
    parser = argparse.ArgumentParser(description='Build compound multi-view dataset.')
    parser.add_argument('--dataset', type=str, default='BDGP', choices=list(data_info.keys()))
    parser.add_argument('--data_model', type=str, default='UI', choices=['UI', 'UN', 'IN', 'UIN'])
    parser.add_argument('--missing_rate', type=float, default=0.5,
                        help='Shared ratio for every active degradation in the compound setting.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--anchor_view', type=int, default=0)
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'salt_pepper', 'dropout'])
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--salt_pepper_ratio', type=float, default=0.1)
    parser.add_argument('--min_noisy_views', type=int, default=1)
    parser.add_argument('--max_noisy_views', type=int, default=None)
    return parser.parse_args()


def getData(name):
    data = []
    LABELS = 0
    np.random.seed(1)
    index = [i for i in range(name['N'])]  # instance number of Dataset
    np.random.shuffle(index)

    Final_data = []
    if name[1] == 'BBCSport':
        data_path = './data/{}.mat'.format(name[1])
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][0][i].T
            diff_view = diff_view.toarray()

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y']) - 1
        LABELS = label[index]

    elif name[1] == 'NUSWIDE':
        data_path = './data/NUSWIDE.mat'

        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X' + str(i + 1)].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'][0])
        unique_elements = []
        seen = set()
        for x in label:
            if x not in seen:
                unique_elements.append(x)
                seen.add(x)

        # 创建映射：元素 -> 索引 (0, 1, 2, ...)
        mapping = {val: idx for idx, val in enumerate(unique_elements)}

        # 替换元素
        result = [mapping[x] for x in label]

        LABELS = result[index]
    elif name[1] == 'Caltech':
        data_path = 'D:\Data_Mining\Code\Datasets\Caltech101-20(.mat)\Caltech101-20.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][i][0].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'])
        LABELS = label[index]
    elif name[1] == 'Scene_15':
        data_path = 'D:\Data_Mining\Code\Datasets\\0000AAA_Other_data\\Scene_15.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][0][i].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'])
        LABELS = label[index]
    elif name[1] == 'LandUse_21':
        data_path = 'D:\Data_Mining\Code\Datasets\LandUse-21\LandUse_21.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][0][i].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'])
        LABELS = label[index]
    elif name[1] == 'HW':
        data_path = './data/HW.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X' + str(i + 1)].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y']).T
        LABELS = label[index]
    elif name[1] == 'Wiki_fea':
        data_path = './data/Wiki_fea.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][i][0].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'])
        LABELS = label[index]
    elif name[1] == 'CUB':
        data_path = 'D:\Data_Mining\Code\Datasets\CUB\cub_googlenet_doc2vec_c10.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][0][i].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['gt'])
        LABELS = label[index]
    elif name[1] == 'CCV':
        data_path = './data/CCV.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][0][i].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'])
        LABELS = label[index]

    elif name[1] == 'PIE_face_10':
        data_path = './data/PIE_face_10.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X'][0][i].T.astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['gt'])
        LABELS = label[index]
    # ---------------------------------- Test Datasets ----------------------------------
    elif name[1] == 'BDGP':
        data_path = 'D:\Data_Mining\Code\Datasets\BDGP\BDGP.mat'
        data = sio.loadmat(data_path)
        for i in range(name['V']):
            diff_view = data['X' + str(i + 1)].astype(np.float32)

            # mm = Normalizer()
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y']).T
        LABELS = label[index]

    elif name[1] == 'NGs':
        data_path = './data/NGs.mat'
        data = sio.loadmat(data_path)

        for i in range(name['V']):
            diff_view = data['X'][i][0].astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y']) - 1
        LABELS = label[index]
        print(LABELS[0])

    elif name[1] == 'Hdigit':
        data_path = 'D:\Data_Mining\Code\Datasets\Hdigit\Hdigit.mat'
        data = sio.loadmat(data_path)

        for i in range(name['V']):
            diff_view = data['data'][0][i].T.astype(np.float32)

            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['truelabel'][0][0]).T - 1
        LABELS = label[index]

    elif name[1] == 'cora':
        data_path = './data/Cora.mat'
        data = sio.loadmat(data_path)
        X = [data['coracites'], data['coracontent']]
        for i in range(name['V']):
            diff_view = X[i].astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['y']) - 1
        LABELS = label[index]
        # print(label.shape)

    elif name[1] == 'cifar10':
        data_path = 'D:\Data_Mining\Code\Datasets\Cifer_10\cifar10.mat'
        data = sio.loadmat(data_path)

        for i in range(name['V']):
            diff_view = data['data'][i][0].T.astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['truelabel'][0][0]) - 1
        LABELS = label[index]
        # print(label.shape)

    elif name[1] == 'stl10_fea':
        file_path = './data/stl10_fea.mat'

        with h5py.File(file_path, 'r') as f:
            # # 查看文件中有哪些键
            # print(list(f.keys()))

            data = f['X']
            refs = np.array(data).flatten()  # shape (5,)

            views = []  # finally, storage datasets and each type of sample is array
            for idx, ref in enumerate(refs):
                # print(f"\n--- Processing view #{idx + 1} ---")
                obj = f[ref]  # 解引用，可能是 Group 或 Dataset
                # print("  HDF5 object type:", type(obj))
                if isinstance(obj, h5py.Group):
                    child_keys = list(obj.keys())
                    # print("  Group keys:", child_keys)

                    ds = obj[child_keys[0]]
                    arr = ds[()]  # 转 numpy
                elif isinstance(obj, h5py.Dataset):
                    arr = obj[()]
                else:
                    raise RuntimeError(f"Unexpected HDF5 type: {type(obj)}")

                # print(f"  view {idx + 1} data shape:", arr.shape)
                views.append(arr.T)

            data = f['Y']
            refs = np.array(data).flatten()  # shape (5,)
            label = refs.T

        for i in views:
            diff_view = i.astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]

            Final_data.append(shuffle_diff_view)

        LABELS = label[index].astype(int)
        LABELS = LABELS.reshape(-1, 1)
        # print(LABELS[0])
        # print(LABELS)
    elif name[1] == 'Reuters':
        data_path = './data/Reuters.mat'
        data = sio.loadmat(data_path)

        for i in range(name['V']):
            diff_view = data['fea'][0][i].toarray().astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['gt']) - 1
        LABELS = label[index]
        # print(label.shape)

    elif name[1] == 'UCI_Digits':
        '''
            contains 2000 instance in 10 clusters for 6 views and feature dimension is [240, 76, 216, 47, 64, 6]
        '''

        data = sio.loadmat('./data/UCI_Digits.mat')

        for i in range(name['V']):
            diff_view = data['fea'][0][i].astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['gt'])
        LABELS = label[index]

    elif name[1] == 'Caltech101_7':

        data = sio.loadmat('data/Caltech101_7.mat')

        for i in range(name['V']):
            diff_view = data['X'][i][0].astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'])
        LABELS = label[index]

    elif name[1] == 'NUSWIDE_deep':
        '''

        '''

        data = sio.loadmat('./data/nuswide_deep_2_view.mat')
        X = ['Img', 'Txt']
        for i in range(name['V']):
            diff_view = data[X[i]].astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['label'].T)
        LABELS = label[index]
    elif name[1] == 'Movies':
        data = sio.loadmat('./data/Movies.mat')
        for i in range(name['V']):
            diff_view = data['X'][i][0].astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['y'])
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    elif name[1] == 'DHA':
        data = sio.loadmat('./data/DHA.mat')
        for i in range(name['V']):
            diff_view = data['X' + str(i + 1)].astype(np.float32)
            # mm = MinMaxScaler()
            # mm = Normalizer()
            mm = StandardScaler()   # best standard method

            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'].T)
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    elif name[1] == 'ALOI':
        data = sio.loadmat('./data/ALOI_100.mat')
        for i in range(name['V']):
            diff_view = data['fea'][0][i].astype(np.float32)
            # mm = MinMaxScaler()
            # mm = Normalizer()
            mm = StandardScaler()

            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = diff_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['gt'])
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    else:
        assert ('No such file or directory')

    return Final_data, LABELS



if __name__ == '__main__':
    args = parse_args()
    dataset = build_dataset_Multi(args)
    print(dataset)
    print('Dataset length:', len(dataset))
    print('First 10 mask rows:')
    print(dataset.mask_matrix[:10])
    if dataset.pair_indices is not None:
        print('First 10 pair-index rows:')
        print(dataset.pair_indices[:10])
    if dataset.noisy_mask is not None:
        print('First 10 noisy-mask rows:')
        print(dataset.noisy_mask[:10])

