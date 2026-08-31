import math
import random

import h5py
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import os
import pandas as pd
import scanpy as sc
import torch.nn.functional as F


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
    NoisyMNIST = {1: 'NoisyMNIST', 'N': 50000, 'K': 10, 'V': 2, 'n_input': [784, 784]},
)


class Incomplete_MultiviewDataset(Dataset):
    def __init__(self, data_list, mask_matrix, labels, num_views):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, num_views, axis=1)

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32) for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False) for v in range(self.num_views)]
        return data, mask, index


def get_mask(num_views, data_size, missing_rate):
    assert num_views >= 2
    miss_sample_num = math.floor(data_size * missing_rate)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, num_views])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(num_views)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[miss_ind[j]] = rand_v
    return mask


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def load_ml_data(args):
    dataset_para = data_info[args.dataset]
    data, labels = getData(dataset_para)


    
    # for ind in range(len(originData['X'])):
    #     curData = np.array(originData['X'][ind])
    #     data.append(curData)


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


def build_dataset(args):
    data_list, labels = load_ml_data(args)
    # if args.norm_type == 'min-max':
    #     data_list = [MinMaxScaler().fit_transform(dv) for dv in data_list]
    # elif args.norm_type == 'standard':
    #     data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    # elif args.norm_type == 'normal':
    #     data_list = [Normalizer().fit(dv).transform(dv) for dv in data_list]
    # else:
    #     pass
    mask = get_mask(args.num_views, args.data_size, args.missing_rate)


    data_list = [data_list[v] * mask[:, v:v + 1] for v in range(args.num_views)]


    incomplete_multiview_dataset = Incomplete_MultiviewDataset(data_list, mask, labels, args.num_views)

    return incomplete_multiview_dataset


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


    elif name[1] == 'NUS_WIDE':

        data_path = './data/NUS_WIDE.mat'

        data = sio.loadmat(data_path)

        # index 也统一转成 numpy int 索引

        index = np.asarray(index, dtype=np.int64).reshape(-1)

        for i in range(name['V']):
            diff_view = data['fea'][0][i].astype(np.float32)

            mm = MinMaxScaler()

            std_view = mm.fit_transform(diff_view)

            shuffle_diff_view = std_view[index]

            Final_data.append(shuffle_diff_view)

        label = np.array(data['gt']).squeeze()

        # 将标签重新映射为 0, 1, 2, ...

        unique_elements = []

        seen = set()

        for x in label:

            if x not in seen:
                unique_elements.append(x)

                seen.add(x)

        mapping = {val: idx for idx, val in enumerate(unique_elements)}

        result = np.array([mapping[x] for x in label], dtype=np.int64)

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
        data_path = './data/Hdigit.mat'
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

        data = sio.loadmat('data/Caltech101-7.mat')

        for i in range(name['V']):
            diff_view = data['X'][i][0].astype(np.float32)
            # print(diff_view.shape)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y'])
        LABELS = label[index]

    elif name[1] == 'Caltech5V':
        mat = sio.loadmat('./data/Caltech-5V.mat')
        '''view_num equip 5'''
        for i in range(5):
            diff_view = mat['X' + str(i + 1)].astype(np.float32)
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)

        Y_list = mat['Y'].T
        label = np.array(Y_list)
        if min(label) == 1:
            label -= -1

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
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['gt'])
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    elif name[1] == 'Reuters_small':
        '''
             contains 1200 instance in 6 clusters for 5 views and feature dimension is [2000, 2000, 2000, 2000, 2000]
         '''
        mat = sio.loadmat('./data/Reuters_small.mat')
        X = mat['fea'][0]
        for i in range(5):
            x = X[i].toarray()
            # mm = StandardScaler()
            mm = MinMaxScaler()
            std_view = mm.fit_transform(x)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.squeeze(mat['gt']).astype('int')
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    # elif name[1] == 'NUS_WIDE':
    #     mat = sio.loadmat('./data/NUS_WIDE.mat')
    #     X = mat['fea'][0]
    #     for i in range(5):
    #         x = X[i].toarray()
    #         # mm = StandardScaler()
    #         mm = MinMaxScaler()
    #         std_view = mm.fit_transform(x)
    #         shuffle_diff_view = std_view[index]
    #         Final_data.append(shuffle_diff_view)
    #     label = np.squeeze(mat['gt']).astype('int')
    #     if np.min(label) == 1:
    #         label -= 1
    #     LABELS = label[index]
    elif name[1] == 'MSRC_v1':
        mat = sio.loadmat('./data/MSRC_v1.mat')
        X = mat['fea'][0]
        for i in range(5):
            x = X[i]
            # mm = StandardScaler()
            mm = MinMaxScaler()
            std_view = mm.fit_transform(x)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.squeeze(mat['gt']).astype('int')
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    elif name[1] == 'flower17':
        mat = sio.loadmat('./data/flower17.mat')
        X = mat['distance_matrices'][0]
        for i in range(name['V']):
            x = X[i]
            # mm = StandardScaler()
            mm = MinMaxScaler()
            std_view = mm.fit_transform(x)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.squeeze(mat['gt']).astype('int')
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    elif name [1] == 'NoisyMNIST':
        mat = sio.loadmat('./data/NoisyMNIST.mat')
        for i in range(name['V']):
            x = mat['X' + str(i + 1)]
            # mm = StandardScaler()
            mm = MinMaxScaler()
            std_view = mm.fit_transform(x)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.squeeze(mat['trainLabel']).astype('int')
        if np.min(label) == 1:
            label -= 1
        LABELS = label[index]
    else:
        assert ('No such file or directory')

    return Final_data, LABELS


if __name__ == '__main__':
    data_name = "BDGP"
    args = parse_args()
    dataset = build_dataset(args)

    print(dataset)











