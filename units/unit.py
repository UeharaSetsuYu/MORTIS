import torch

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F
# split X to Complete data and Incomplete data.
import seaborn as sns
EPSILON = 1e-8
def split_complete_incomplete(X, mask=None):
    """
    将多视图数据 X 拆分为完整数据子集 X1 和不完整数据子集 X2。
    (已更新为纯 PyTorch 实现，完美支持 GPU Tensor 哦)
    """
    num_views = len(X)
    num_samples = X[0].shape[0]

    # 如果没有传 mask，就只好靠自己用 PyTorch 去推断了呢
    if mask is None:
        # 获取当前数据所在的设备 (CPU 或 CUDA)
        device = X[0].device
        mask = torch.zeros((num_samples, num_views), device=device)
        for v in range(num_views):
            # 用 torch 替代 np
            is_observed = (torch.sum(torch.abs(X[v]), dim=1) > 1e-8).int()
            mask[:, v] = is_observed

    # 计算每个样本拥有的视图总数
    mask_sum = torch.sum(mask, dim=1)

    # 用 torch.where 替代 np.where
    complete_idx = torch.where(mask_sum == num_views)[0]
    incomplete_idx = torch.where(mask_sum < num_views)[0]

    X1 = [view_data[complete_idx] for view_data in X]
    X2 = [view_data[incomplete_idx] for view_data in X]

    return X1, X2, complete_idx, incomplete_idx



# excute K-Means to clustering and alignment
def kmeans_with_alignment(latent_fusion, y_true):
    # 无论是 Tensor 还是普通的 List，都温柔地把大家统一变成 Numpy 数组
    if hasattr(latent_fusion, 'cpu'):
        latent_fusion = latent_fusion.cpu().detach().numpy()
    else:
        latent_fusion = np.array(latent_fusion)

    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().detach().numpy()
    else:
        y_true = np.array(y_true)

    # 【就是这里哦！】确保大家是一维的，并且强迫大家变成干干净净的整数
    y_true = np.squeeze(y_true).astype(int)

    # 看看大家到底原本分成了几个群组
    n_clusters = len(np.unique(y_true))

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    y_pred_raw = kmeans.fit_predict(latent_fusion)

    D = max(y_pred_raw.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # 这次大家都是整数了，绝对不会再在这里报错了呢
    for i in range(y_pred_raw.size):
        w[y_pred_raw[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([mapping[val] for val in y_pred_raw])

    return y_pred_aligned

# ELBO
def gaussian_kl(q_mu, q_var, p_mu=None, p_var=None):
    if p_mu is None:
        p_mu = torch.zeros_like(q_mu)
    if p_var is None:
        p_var = torch.ones_like(q_var)
    kl = - 0.5 * (torch.log(q_var / p_var) - q_var / p_var - torch.pow(q_mu - p_mu, 2) / p_var + 1)
    return kl.sum(-1).mean()


class SinkhornDistance(nn.Module):
    """
    计算两个不同规模样本集之间的 Wasserstein 距离（Sinkhorn 近似）。
    适用于输入尺寸为 (m1, d) 和 (m2, d) 的情况。
    """

    def __init__(self, eps=0.1, max_iter=100, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # x: (m1, d), y: (m2, d)
        m1 = x.size(0)
        m2 = y.size(0)
        device = x.device

        # 1. 计算代价矩阵 (Cost Matrix) C
        # C[i, j] 表示 x[i] 和 y[j] 之间的平方欧式距离
        x_col = x.unsqueeze(1)  # (m1, 1, d)
        y_lin = y.unsqueeze(0)  # (1, m2, d)
        C = torch.sum((x_col - y_lin) ** 2, dim=-1)  # (m1, m2)

        # 2. 初始化经验分布权重 (Empirical measures)
        # 因为尺寸不一，我们假设每个样本的权重是均匀的
        mu = torch.empty(m1, dtype=x.dtype, device=device).fill_(1.0 / m1)
        nu = torch.empty(m2, dtype=y.dtype, device=device).fill_(1.0 / m2)

        # 3. Sinkhorn 迭代过程
        # K 是 Gibbs 核
        K = torch.exp(-C / self.eps)
        u = torch.ones_like(mu)

        for _ in range(self.max_iter):
            # 更新变量 v
            v = nu / (torch.matmul(K.t(), u.unsqueeze(1)).squeeze(1) + 1e-8)
            # 更新变量 u
            u = mu / (torch.matmul(K, v.unsqueeze(1)).squeeze(1) + 1e-8)

        # 4. 计算最优传输平面 (Optimal Transport Plan) P 和 距离
        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        wasserstein_dist = torch.sum(P * C)

        if self.reduction == 'mean':
            return wasserstein_dist
        return wasserstein_dist, P






def topk_gumbel_softmax(k, logits, tau, hard=True):
    """
    Applies the top-k Gumbel-Softmax operation to the input logits.

    Args:
        k (int): The number of elements to select from the logits.
        logits (torch.Tensor): The input logits.
        tau (float): The temperature parameter for the Gumbel-Softmax operation.
        hard (bool, optional): Whether to use the straight-through approximation.
            If True, the output will be a one-hot vector. If False, the output will be a
            continuous approximation of the top-k elements. Default is True.

    Returns:
        torch.Tensor: The output tensor after applying the top-k Gumbel-Softmax operation.
    """
    m = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    g = m.sample()
    logits = logits + g

    # continuous top k
    khot = torch.zeros_like(logits).type_as(logits)
    onehot_approx = torch.zeros_like(logits).type_as(logits)
    for i in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).type_as(logits))
        logits = logits + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(logits / tau, dim=1)
        khot = khot + onehot_approx

    if hard:
        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot
    return res




class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def GRL(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


import pandas as pd


def save_lists_to_excel(list1, list2, list3, list4, Data_name, filename = '_Clu_Performance'):
    """
    将 4 个列表保存到同一个 Excel 文件中。 ACC, ARI, NMI, PUR, respectively
    """
    # 将列表转换为 pd.Series，这能完美解决列表长度不一致导致无法创建 DataFrame 的问题
    data = {
        "ACC": pd.Series(list1),
        "ARI": pd.Series(list2),
        "NMI": pd.Series(list3),
        "PUR": pd.Series(list4)
    }

    df = pd.DataFrame(data)
    filename = 'Result_Doc/' + Data_name + filename + ".xlsx"
    # 将 DataFrame 保存为 Excel 文件，index=False 表示不保存行索引
    df.to_excel(filename, sheet_name='Result', index=False)
    print(f"Result Save：{filename}")


def save_Loss_to_excel(list1, list2, list3, list4, Data_name):
    """
    将 4 个列表保存到同一个 Excel 文件中。 Loss_All, Loss_rec, Loss_Causal, Loss_Reg, respectively
    """
    # 将列表转换为 pd.Series，这能完美解决列表长度不一致导致无法创建 DataFrame 的问题
    data = {
        "ACC": pd.Series(list1),
        "ARI": pd.Series(list2),
        "NMI": pd.Series(list3),
        "PUR": pd.Series(list4)
    }

    df = pd.DataFrame(data)
    filename = 'Result_Doc/' + Data_name + "_Clu_Performance.xlsx"
    # 将 DataFrame 保存为 Excel 文件，index=False 表示不保存行索引
    df.to_excel(filename, index=False)
    print(f"Result Save：{filename}")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_data_distributions(X1: torch.Tensor, X2: torch.Tensor,
                            labels=('X1', 'X2')):
    """
    可视化两个不同维度 Tensor (N, D1) 和 (N, D2) 的数据分布。
    包含全局数值分布与样本特征大小(L2 Norm)分布。
    """
    # 乖乖把它们从 GPU 拿下来，然后转成 numpy，这步是不能省的哦
    x1_np = X1.detach().cpu().numpy()
    x2_np = X2.detach().cpu().numpy()

    # 计算每一行（每一个样本）的 L2 范数（大小）
    # 因为维度不同，直接比均值可能会有偏差，比向量的整体大小会更合理一些
    x1_norms = np.linalg.norm(x1_np, axis=1)
    x2_norms = np.linalg.norm(x2_np, axis=1)

    # 展平数据，用于看全局的数值分布
    x1_flat = x1_np.flatten()
    x2_flat = x2_np.flatten()

    # 准备画图啦，这次给你画两个子图并排哦
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左图：全局数值的密度分布 ---
    # 使用 KDE (核密度估计) 会比方块状的直方图看起来更平滑、更漂亮
    sns.kdeplot(x1_flat, fill=True, color='skyblue', label=f'{labels[0]} (Dim={x1_np.shape[1]})', ax=axes[0])
    sns.kdeplot(x2_flat, fill=True, color='lightcoral', label=f'{labels[1]} (Dim={x2_np.shape[1]})', ax=axes[0])
    axes[0].set_title('Global Value Distribution (Flattened)')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- 右图：样本向量大小(L2 Norm)的分布 ---
    sns.kdeplot(x1_norms, fill=True, color='dodgerblue', label=labels[0], ax=axes[1])
    sns.kdeplot(x2_norms, fill=True, color='crimson', label=labels[1], ax=axes[1])
    axes[1].set_title('Row-wise L2 Norm Distribution')
    axes[1].set_xlabel('L2 Norm Magnitude')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def print_missing_rates(X):
    """
    计算多视图数据列表中，每个视图的数据丢失率（全 0 向量的占比）。
    """
    # num_samples = X[0].shape[0]
    output_strings = []

    for v, view_data in enumerate(X):
        # 找出那些特征绝对值之和极其接近 0 的样本，判定为数据丢失 (0向量)
        is_missing = (torch.sum(torch.abs(view_data), dim=1) < 1e-8)

        # 计算丢失率
        missing_rate = is_missing.float().mean().item()

        # 按照要求的英文格式准备输出字符串
        output_strings.append(f"View {v + 1} data missing rate: {missing_rate:.2%}")

    # 将所有视图的丢失率拼接成一行并打印
    print(", ".join(output_strings))


import torch


def mahalanobis_distance(X1, X2, eps=1e-5):
    """
    计算两个特征矩阵 X1 和 X2 对应样本之间的马氏距离。

    参数:
    X1: tensor, 形状为 [Batch_size, Feature_dim] (例如：不完备特征 e_incomp)
    X2: tensor, 形状为 [Batch_size, Feature_dim] (例如：完备特征 e_comp，作为目标分布)
    eps: float, 保证协方差矩阵可逆的微小扰动值

    返回:
    mahalanobis_dist: tensor, 形状为 [Batch_size]，每个样本对应的马氏距离
    """
    # 那个... 首先要确保它们的形状是一致的哦
    if X1.shape != X2.shape:
        raise ValueError("X1 和 X2 的形状必须完全一样呢...")

    batch_size, feature_dim = X1.shape

    # 1. 计算目标分布（完备特征 X2）的协方差矩阵
    # 注意：torch.cov 期望的输入形状是 [Feature_dim, Batch_size]，所以我们要对 X2 转置一下
    cov_matrix = torch.cov(X2.T)

    # 2. 加上微扰项，防止矩阵奇异（不可逆）导致训练直接崩溃
    identity = torch.eye(feature_dim, device=X2.device)
    cov_matrix_stable = cov_matrix + eps * identity

    # 3. 计算协方差矩阵的逆
    # 如果特征维度 d 特别大，这一步可能会稍微有些慢哦
    inv_cov_matrix = torch.linalg.inv(cov_matrix_stable)

    # 4. 计算两个特征之间的差异
    delta = X1 - X2

    # 5. 计算马氏距离的平方
    # 对应数学公式: (X1 - X2)^T * Sigma^-1 * (X1 - X2)
    # 利用矩阵乘法和逐元素相乘，可以一次性算出整个 Batch 的结果
    left_term = torch.matmul(delta, inv_cov_matrix)
    mahalanobis_sq = torch.sum(left_term * delta, dim=1)

    # 6. 开根号得到最终距离
    # 使用 clamp 限制最小值，是为了防止浮点数精度误差产生极小的负数，导致 sqrt 算出 NaN
    mahalanobis_dist = torch.sqrt(torch.clamp(mahalanobis_sq, min=1e-8))

    return mahalanobis_dist


# === 如果你的 Batch Size 小于特征维度（比如 BS=128, Dim=512） ===
# === 请务必使用下面这个对角近似的版本哦 ===
def diagonal_mahalanobis_distance(X1, X2, eps=1e-5):
    """
    仅使用方差（协方差矩阵的对角线）进行近似的马氏距离。
    计算极其迅速，且完全不用担心矩阵求逆报错呢。
    """
    delta = X1 - X2

    # 仅计算 X2 在每个维度上的方差 [Feature_dim]
    var = torch.var(X2, dim=0, unbiased=True)
    var_stable = var + eps

    # 相当于乘以对角矩阵的逆
    mahalanobis_sq = torch.sum((delta ** 2) / var_stable, dim=1)

    return torch.sqrt(torch.clamp(mahalanobis_sq, min=1e-8))


def get_transition_probabilities(X: torch.Tensor, tau: float = 1.0, mask_diag: bool = True,
                                 eps: float = 1e-5) -> torch.Tensor:
    """
    计算样本间的软性转移概率矩阵（基于全协方差马氏距离）。
    """
    if X.dim() == 1:
        X = X.unsqueeze(1)

    N, d = X.shape

    # ==========================================
    # 1. 计算两两之间的马氏距离 (Pairwise Mahalanobis)
    # ==========================================
    # 计算协方差矩阵并加微扰
    cov = torch.cov(X.T)
    cov_stable = cov + eps * torch.eye(d, device=X.device)

    # 求协方差矩阵的逆
    inv_cov = torch.linalg.inv(cov_stable)

    # 技巧：计算 X * Sigma^-1，形状为 [N, d]
    X_inv_cov = torch.matmul(X, inv_cov)

    # 计算所有交叉项 X_i * Sigma^-1 * X_j^T，形状为 [N, N]
    cross_term = torch.matmul(X_inv_cov, X.T)

    # 提取对角线元素，也就是 X_i * Sigma^-1 * X_i^T，形状为 [N]
    diag_term = torch.diagonal(cross_term)

    # 利用广播机制拼装成完整公式: a^2 + b^2 - 2ab
    # unsqueeze(1) 变列向量 [N, 1]，unsqueeze(0) 变行向量 [1, N]
    dist_sq = diag_term.unsqueeze(1) + diag_term.unsqueeze(0) - 2 * cross_term

    # 开根号得到 N x N 的马氏距离矩阵
    dist_matrix = torch.sqrt(torch.clamp(dist_sq, min=1e-8))

    # ==========================================
    # 2. 转换为概率分布
    # ==========================================
    logits = -dist_matrix / tau

    # logits = torch.exp(logits)

    if mask_diag and logits.dim() >= 2:
        logits.fill_diagonal_(float('-inf'))

    Y = F.softmax(logits, dim=1)

    return Y


# --- Lin 可以这样测试一下哦 ---

def GRL_coeff(epoch, beta, n):
    coeff = 2 / (1 + np.exp(-beta * epoch / n)) - 1
    return coeff


import torch
import torch.nn as nn
import torch.nn.functional as F


def target_distribution(q):
    """
    根据软分配概率 Q 计算目标分布 P
    q: 形状为 (Batch_size, Num_clusters) 的张量
    """
    # 1. 计算 q 的平方
    weight = q ** 2

    # 2. 计算每个簇的软频率 (按列求和)
    # 加上 1e-8 是为了防止除以零的崩溃哦
    cluster_freq = weight.sum(dim=0) + 1e-8

    # 3. 平方项除以频率
    weight = weight / cluster_freq

    # 4. 归一化，使得每个样本属于各簇的概率和为 1
    p = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)

    # 使用 detach() 很关键！P 是我们的“目标”，不应该参与梯度反向传播
    return p.detach()


def dec_kl_loss(z, cluster_centers, alpha=1.0):
    """
    计算 DEC 的 KL 散度损失
    z: 网络输出的表征 (Batch_size, Feature_dim)
    cluster_centers: 聚类中心参数 (Num_clusters, Feature_dim)
    """
    # 1. 计算 z 和聚类中心之间的欧氏距离的平方
    # z.unsqueeze(1): (B, 1, D)
    # cluster_centers.unsqueeze(0): (1, K, D)
    dist = torch.sum((z.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, dim=2)

    # 2. 使用 t-分布计算软分配概率 Q
    q = 1.0 / (1.0 + dist / alpha)
    # 按照上面公式的指数，如果 alpha=1，这里就是 q 的一次方
    q = q ** ((alpha + 1.0) / 2.0)
    # 归一化得到最终的 Q 分布
    q = q / torch.sum(q, dim=1, keepdim=True)

    # 3. 计算目标分布 P
    p = target_distribution(q)

    # 4. 计算 KL 散度
    # PyTorch 的 kl_div 默认输入要求是 log 概率，目标是普通概率
    # reduction='batchmean' 是标准的数学定义行为
    loss = F.kl_div(q.log(), p, reduction='batchmean')

    return loss, q, p


def student_q(z, cluster_centers, alpha = 1.0):
    dist = torch.sum((z.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, dim=2)
    q = 1.0 / (1.0 + dist / alpha)
    # 按照上面公式的指数，如果 alpha=1，这里就是 q 的一次方
    q = q ** ((alpha + 1.0) / 2.0)
    # 归一化得到最终的 Q 分布
    q = q / torch.sum(q, dim=1, keepdim=True)

    return q


def view_guided_common_loss(q_common, p_views, view_weights=None):
    """
    使用视图特定的聚类分布，来指导不变特征的聚类语义

    参数:
        q_common: 不变特征(公共空间)的软概率分布 (Batch_size, Num_clusters)
        p_views: 一个列表，包含各个特定视图提纯后的目标分布 P_v
                 每个 P_v 必须是已经 .detach() 过的！
        view_weights: (可选) 各个视图的可靠性权重，默认为均等权重
    """
    num_views = len(p_views)
    if view_weights is None:
        view_weights = [1.0 / num_views] * num_views

    loss = 0.0
    # q_common 是“学生”，计算它的对数概率
    q_common_log = torch.log(q_common + 1e-8)

    for v in range(num_views):
        p_v = p_views[v]  # 视图 v 的目标分布 (老师)
        weight = view_weights[v]

        # 让公共空间向视图 v 学习
        # KL(P_v || Q_common) -> F.kl_div(student_log, teacher)
        view_loss = F.kl_div(q_common_log, p_v, reduction='batchmean')

        loss += weight * view_loss

    return loss


def get_transition_prob(Z_v, mask_v, tau=1.0):
    """
    计算单个视图的转移概率矩阵 P

    参数:
    Z_v: (Batch, dim) 当前视图的表示矩阵
    mask_v: (Batch, 1) 或 (Batch,) 当前视图的缺失掩码，1表示存在，0表示缺失
    tau: 温度系数，控制概率分布的平滑程度
    """
    # 确保掩码是一维的 (Batch,)，方便后续操作
    mask_v = mask_v.squeeze()
    Z_normalized = F.normalize(Z_v, p=2, dim=1)
    # 1. 计算两两之间的相似度 (这里使用的是点积，你也可以换成负欧式距离)
    sim_matrix = torch.matmul(Z_normalized, Z_normalized.T) / tau

    # 2. 构建二维联合掩码 (Batch, Batch)
    # 只有当样本 i 和样本 j 都真实存在时，mask_2d[i, j] 才是 1
    mask_2d = mask_v.unsqueeze(1) * mask_v.unsqueeze(0)

    # 3. 施加“致命的掩码惩罚” (在 Softmax 之前)
    # 首先，绝对不能把概率转移给自己哦，所以对角线要屏蔽掉
    sim_matrix.fill_diagonal_(-1e9)
    # 然后，把所有和缺失样本（幻影）相关的相似度全部降到极低
    sim_matrix = sim_matrix.masked_fill(mask_2d == 0, -1e9)

    # 4. 计算转移概率
    # 因为前面填充了 -1e9，转向缺失样本的 exp() 结果会无限趋近于 0
    P_view = F.softmax(sim_matrix, dim=1)

    # 5. 彻底的清理工作 (在 Softmax 之后)
    # 对于那些原本就缺失的样本，PyTorch 的 softmax 会给整行输出均匀分布
    # 我们必须把这些本不该存在的行全部强行归零，防止它们产生错误的梯度
    P_view = P_view * mask_v.unsqueeze(1)

    return P_view



L2norm = nn.functional.normalize

def kernel_affinity(z, temperature=0.05, step: int = 5, alpha = 0.5):
    z = L2norm(z)
    G = (2 - 2 * (z @ z.t())).clamp(min=0.)
    G = torch.exp(-G / temperature)
    G = G / G.sum(dim=1, keepdim=True)

    G = torch.matrix_power(G, step)

    G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)

    return G



def compute_gromov_wasserstein(X, Y, epsilon=1e-2):
    """
    计算两个视图特征 X 和 Y 之间的 Gromov-Wasserstein 距离。

    参数:
        X: torch.Tensor, shape (n, d_x), 位于 GPU
        Y: torch.Tensor, shape (m, d_y), 位于 GPU
        epsilon: float, 熵正则化系数 (推荐开启，能让梯度回传更平滑，计算更快)

    返回:
        gw_dist: torch.Tensor, 标量, 两个分布之间的 GW 距离
    """
    # 确保输入是 float 类型，避免距离计算时精度溢出
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)

    n = X.size(0)
    m = Y.size(0)

    # 1. 计算各自视图内部的距离矩阵 (Cost matrices)
    # 这里使用 L2 距离。因为两个视图的维度 d_x 和 d_y 可能不同，
    # 所以我们不直接跨视图算距离，而是算它们各自的内部结构
    C_X = torch.cdist(X, X, p=2)
    C_Y = torch.cdist(Y, Y, p=2)

    # 【重要提醒】为了数值稳定，建议对距离矩阵进行归一化
    # 否则不同视图的特征尺度差异过大，很容易导致 GW 算出的梯度爆炸或者变为 NaN 呢……
    C_X = C_X / (C_X.max() + 1e-8)
    C_Y = C_Y / (C_Y.max() + 1e-8)

    # 2. 定义边缘分布 (Marginal distributions)
    # 这里我们假设视图内部每个样本的权重是均匀的
    p = torch.ones(n, device=X.device, dtype=X.dtype) / n
    q = torch.ones(m, device=Y.device, dtype=Y.dtype) / m

    # 3. 计算 GW 距离
    # 使用 entropic_gromov_wasserstein2 直接返回 loss 标量
    # 只要传入的是 PyTorch GPU Tensors，POT 会自动切入 PyTorch 后端处理
    if epsilon > 0:
        # 加上熵正则化，Sinkhorn 迭代在 GPU 上跑得会非常快哦
        gw_dist = ot.gromov.entropic_gromov_wasserstein2(
            C_X, C_Y, p, q, loss_fun='square_loss', epsilon=epsilon
        )
    else:
        # 如果 Lin 必须要求精确的 GW 距离，可以设 epsilon=0
        # 但在深度学习训练里，精确解的梯度有时候会有些突兀……
        gw_dist = ot.gromov.gromov_wasserstein2(
            C_X, C_Y, p, q, loss_fun='square_loss'
        )

    return gw_dist



def GetSimilarity(fea_mat1, fea_mat2):
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
    return Sim_mat


def MissingStatic(dataset, view_num):
    # ===== Missing statistics for each view by true label =====
    labels = np.asarray(dataset.labels).squeeze()
    mask_matrix = np.concatenate(dataset.mask_list, axis=1)  # shape: [N, view_num]
    unique_labels = np.unique(labels)

    print("============= Missing Statistics by View and Label =============")
    for v in range(view_num):
        view_mask = mask_matrix[:, v]  # 1: observed, 0: missing
        missing_num = int((view_mask == 0).sum())
        observed_num = int((view_mask == 1).sum())
        print(f"View {v + 1}: observed={observed_num}, missing={missing_num}, "
              f"missing_rate={missing_num / len(view_mask):.4f}")

        for c in unique_labels:
            cls_idx = (labels == c)
            cls_total = int(cls_idx.sum())
            cls_missing = int((view_mask[cls_idx] == 0).sum())
            cls_observed = int((view_mask[cls_idx] == 1).sum())
            cls_missing_rate = cls_missing / cls_total if cls_total > 0 else 0.0

            print(f"  Label {c}: observed={cls_observed}, missing={cls_missing}, "
                  f"total={cls_total}, missing_rate={cls_missing_rate:.4f}")
    print("===============================================================")


def compute_global_affinity_single(Q):
    """
    根据公式 (12) 计算全局归一化亲和力矩阵 A
    :param Q: 形状为 (K, d) 的 GPU Tensor
    :return: 形状为 (K, K) 的亲和力矩阵 A (同样位于 GPU)
    """
    # 1. 计算 Q_i 乘 (Q_j)^T，得到内积相似度矩阵
    # 结果 M 是一个 (K, K) 的矩阵
    M = torch.matmul(Q, Q.T)

    # --- 数值稳定性防溢出处理 ---
    # 在 GPU 上算指数，如果 M 里的数值太大很容易变成 inf (无穷大)。
    # 减去最大值是一个非常常用的技巧，完全不会改变最终归一化后的数学结果呢。
    M_max = torch.max(M)

    # 2. 计算分子：e 的指数
    exp_M = torch.exp(M - M_max)

    # 3. 计算分母：矩阵中所有元素的总和
    # (完美对应了公式底下的双重求和符号 sum_t sum_l)
    denominator = torch.sum(exp_M)

    # 4. 计算最终的矩阵 A
    A = exp_M / denominator

    return A

def evaluate_prototypes_with_data(P, C, labels=None, eps=1e-8, temperature=0.1):
    """
    Evaluate prototype quality given prototypes P and sample features C.

    Args:
        P: torch.Tensor or array-like, shape [K, d]
            Prototype matrix.
        C: torch.Tensor or array-like, shape [N, d]
            Sample features.
        labels: optional, shape [N]
            Ground-truth labels, used only for purity if provided.
        eps: numerical stability.
        temperature: temperature for soft assignment.

    Returns:
        metrics: dict
    """
    P = torch.as_tensor(P, dtype=torch.float32)
    C = torch.as_tensor(C, dtype=torch.float32)

    if P.dim() != 2 or C.dim() != 2:
        raise ValueError(f"P and C must be 2D tensors, got P={tuple(P.shape)}, C={tuple(C.shape)}")
    if P.size(1) != C.size(1):
        raise ValueError(f"Feature dim mismatch: P has {P.size(1)}, C has {C.size(1)}")

    K, d = P.shape
    N = C.shape[0]

    # Normalize
    Pn = F.normalize(P, p=2, dim=1)
    Cn = F.normalize(C, p=2, dim=1)

    # Sample-prototype cosine similarity: [N, K]
    sim = Cn @ Pn.T

    # Hard assignment
    top2 = torch.topk(sim, k=min(2, K), dim=1)
    assign = top2.indices[:, 0]
    max_sim = top2.values[:, 0]
    second_sim = top2.values[:, 1] if K > 1 else torch.zeros_like(max_sim)

    # 1) Compactness: how close samples are to their assigned prototype
    compactness = max_sim.mean().item()

    # 2) Margin: how much better top-1 is than top-2
    margin = (max_sim - second_sim).mean().item() if K > 1 else 0.0

    # 3) Within-cluster scatter: lower is better
    scatter_list = []
    for k in range(K):
        idx = (assign == k)
        if idx.any():
            scatter_list.append((1.0 - (Cn[idx] @ Pn[k].unsqueeze(1)).squeeze(1)).mean())
    within_scatter = torch.stack(scatter_list).mean().item() if scatter_list else 0.0

    # 4) Prototype separation
    S = Pn @ Pn.T
    offdiag_mask = ~torch.eye(K, dtype=torch.bool, device=P.device)
    offdiag = S[offdiag_mask]
    avg_offdiag_cosine = offdiag.mean().item() if offdiag.numel() > 0 else 0.0
    max_offdiag_cosine = offdiag.max().item() if offdiag.numel() > 0 else 0.0
    ortho_err = (torch.norm(S - torch.eye(K, device=P.device), p="fro") / max(K, 1)).item()

    # 5) Assignment balance
    hist = torch.bincount(assign, minlength=K).float()
    p_assign = hist / (hist.sum() + eps)
    uniform = torch.full_like(p_assign, 1.0 / K)
    balance_kl = F.kl_div((p_assign + eps).log(), uniform, reduction="sum").item()
    assignment_entropy = -(p_assign * (p_assign + eps).log()).sum().item()
    effective_prototypes = torch.exp(torch.tensor(assignment_entropy)).item()  # higher is better
    used_ratio = (hist > 0).float().mean().item()

    # 6) Soft assignment entropy (confidence, but too low may collapse)
    soft_q = F.softmax(sim / temperature, dim=1)
    soft_entropy = -(soft_q * (soft_q + eps).log()).sum(dim=1).mean().item()

    # 7) Spectral diagnostics of prototypes
    svals = torch.linalg.svdvals(Pn).clamp_min(eps)
    prob = svals / svals.sum()
    effective_rank = torch.exp(-(prob * (prob + eps).log()).sum()).item()
    condition_number = (svals.max() / svals.min()).item()

    # 8) Optional purity if labels are available
    purity = None
    if labels is not None:
        labels = torch.as_tensor(labels).view(-1)
        if labels.numel() != N:
            raise ValueError(f"labels length mismatch: got {labels.numel()}, expected {N}")
        purity_scores = []
        for k in range(K):
            idx = (assign == k)
            if idx.any():
                lab = labels[idx]
                counts = torch.bincount(lab.long())
                purity_scores.append((counts.max().float() / counts.sum().float()).item())
        purity = sum(purity_scores) / len(purity_scores) if purity_scores else 0.0

    # A simple composite score in [roughly 0, 1], higher is better.
    # Tune weights if you care more about separation or balance.
    diversity_score = max(0.0, min(2.0, 1.0 - avg_offdiag_cosine)) / 2.0
    compact_score = max(0.0, min(1.0, (compactness + 1.0) / 2.0))
    margin_score = max(0.0, min(1.0, (margin + 1.0) / 2.0))
    balance_score = max(0.0, min(1.0, 1.0 - balance_kl / np.log(float(K) + 1e-8)))
    rank_score = max(0.0, min(1.0, effective_rank / float(K)))

    quality_score = (
        0.25 * compact_score +
        0.20 * margin_score +
        0.20 * diversity_score +
        0.20 * balance_score +
        0.15 * rank_score
    )

    metrics = {
        "num_prototypes": K,
        "dim": d,
        "num_samples": N,
        "compactness_mean_cos": compactness,
        "margin_top1_top2": margin,
        "within_scatter": within_scatter,
        "avg_offdiag_cosine": avg_offdiag_cosine,
        "max_offdiag_cosine": max_offdiag_cosine,
        "orthogonality_error": ortho_err,
        "assignment_entropy": assignment_entropy,
        "soft_assignment_entropy": soft_entropy,
        "balance_kl_to_uniform": balance_kl,
        "effective_prototypes": effective_prototypes,
        "used_ratio": used_ratio,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "quality_score": quality_score,
    }

    if purity is not None:
        metrics["prototype_purity"] = purity

    return metrics


def compute_A_hat(Q_hat_v):
    """
    Q_hat_v: Tensor, shape [K, D]
        第 v 个视图下的 K 个原型/簇表示。

    return:
        A_hat: Tensor, shape [K, K]
        公式中的 \hat{A}^v
    """
    # pairwise dot product: [K, K]
    sim = torch.matmul(Q_hat_v, Q_hat_v.T)

    # exp(Q_i Q_j^T)
    exp_sim = torch.exp(sim)

    # denominator: sum_{t=1}^K sum_{l=1}^K exp(Q_t Q_l^T)
    denom = exp_sim.sum()

    # A_ij
    A_hat = exp_sim / (denom + 1e-8)

    return A_hat

import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_comprehensive_metrics(P, Z):
    """
    计算综合聚类指标 (Silhouette, DBI, CHI)
    P: shape [K, d], 原型矩阵 (GPU Tensor)
    Z: shape [N, d], 样本特征矩阵 (GPU Tensor)
    """
    # 确保它们都是 Tensor，并且没有多余的梯度阻碍计算呢
    with torch.no_grad():
        # 首先，要把大家（Z）好好地分配给对应的圈子（原型 P）
        # 这里我用了欧氏距离来衡量远近哦
        dist = torch.cdist(Z, P)  # 计算距离矩阵 [N, K]
        labels = torch.argmin(dist, dim=1)  # 找到最近的原型作为标签

        # 为了稳妥地计算指标，稍微挪动一下位置……转到 CPU 和 numpy
        Z_np = Z.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

    # 要确认一下是不是至少分成了两个圈子呢。
    # 如果所有人都只挤在一个原型里，系统就崩溃了，这些指标也失去了意义……
    unique_labels = np.unique(labels_np)

    if len(unique_labels) > 1:
        # 1. 轮廓系数 (越接近 1 越完美，说明大家内聚且排他)
        sil = silhouette_score(Z_np, labels_np)

        # 2. 戴维斯-布尔丁指数 (越小越好，说明圈子之间互不干扰)
        dbi = davies_bouldin_score(Z_np, labels_np)

        # 3. 卡林斯基-哈拉巴斯指数 (越大越好，说明各个圈子界限分明像孤岛一样)
        chi = calinski_harabasz_score(Z_np, labels_np)
    else:
        # 如果只剩下一个孤零零的圈子，那只能给出最差的分数了……我不想看到那种情况呢。
        sil = -1.0
        dbi = float('inf')
        chi = 0.0

    return {
        "Silhouette_Coefficient": float(sil),
        "Davies_Bouldin_Index": float(dbi),
        "Calinski_Harabasz_Index": float(chi)
    }

''' Ma-based evaluation of Prototype'''
class Ma_Proto_Eva(nn.Module):
    def __init__(self):
        super(Ma_Proto_Eva, self).__init__()
    def forward(self, Q, P, C):
        Compact, *_ = self.weighted_diag_mahalanobis_compact(Q, C, P)
        Separation, _ = self.diag_mahalanobis_separation(Q, C)
        Ma_Score = Compact / (Separation + 1e-8)
        return Ma_Score

    def weighted_diag_mahalanobis_compact(self, Q, C, P, eps=1e-8):
        """
        Q: Tensor, shape [K, D]
           prototype matrix

        C: Tensor, shape [N, D]
           sample feature matrix

        P: Tensor, shape [N, K]
           cluster assignment probability / enhanced assignment probability

        return:
            compact: scalar Tensor
                mean weighted compactness over prototypes

            compact_per_proto: Tensor, shape [K]
                weighted compactness for each prototype

            dist2: Tensor, shape [N, K]
                diagonal Mahalanobis squared distance between each sample and prototype

            weighted_dist2: Tensor, shape [N, K]
                P-weighted diagonal Mahalanobis squared distance
        """
        var = C.var(dim=0, unbiased=False) + eps  # [D]

        diff = C.unsqueeze(1) - Q.unsqueeze(0)  # [N, K, D]

        dist2 = (diff.pow(2) / var.view(1, 1, -1)).sum(dim=2)  # [N, K]

        P = P / (P.sum(dim=1, keepdim=True) + eps)  # [N, K]

        weighted_dist2 = P * dist2  # [N, K]

        compact_per_proto = weighted_dist2.sum(dim=0) / (P.sum(dim=0) + eps)  # [K]

        compact = compact_per_proto.mean()

        return compact, compact_per_proto, dist2, weighted_dist2

    def diag_mahalanobis_separation(self, Q, C, eps=1e-8, sqrt=False):
        """
        Compute prototype separation with diagonal Mahalanobis distance.

        Q: Tensor, shape [K, D]
           Prototype matrix for one view.

        C: Tensor, shape [N, D]
           Feature matrix for the same view.
           Used to estimate diagonal covariance.

        eps:
           Numerical stability term.

        sqrt:
           If False, return squared Mahalanobis distance.
           If True, return Mahalanobis distance.

        return:
           separation: scalar Tensor
               Minimum pairwise prototype distance.

           dist_matrix: Tensor, shape [K, K]
               Pairwise prototype distance matrix.
        """
        K, D = Q.shape

        var = C.var(dim=0, unbiased=False).clamp_min(eps)  # [D]

        diff = Q.unsqueeze(1) - Q.unsqueeze(0)  # [K, K, D]

        dist2 = (diff.pow(2) / var.view(1, 1, -1)).sum(dim=2)  # [K, K]

        eye = torch.eye(K, device=Q.device, dtype=torch.bool)

        dist2_no_diag = dist2.masked_fill(eye, float("inf"))

        separation = dist2_no_diag.min()

        if sqrt:
            dist = torch.sqrt(dist2.clamp_min(eps))
            separation = torch.sqrt(separation.clamp_min(eps))
            return separation, dist

        return separation, dist2


class ProtoFeatureLoss(nn.Module):
    def __init__(self):
        super(ProtoFeatureLoss, self).__init__()
        self.EPS = 1e-8
    def forward(self, C, Q, P):
        A, _ = self.compute_attention_A(P, Q, C)
        dist = self.diag_mahalanobis_distance(C, Q)
        KL_loss = F.kl_div(
            torch.log(P.clamp_min(self.EPS)),
            A.detach(),
            reduction='batchmean'
        )
        loss_compact = (A.detach() * dist).sum(dim=1).mean()
        loss = KL_loss + 0.5 * loss_compact
        return loss

    def forward_Attention(self, model, C, Q, P_prior, P_target=None):
        """
        P_prior: [N, K], usually the enhanced assignment P_tilde used to compute A.
        P_target: [N, K], usually the raw cluster_head output P supervised by A.
        """
        Q_safe = Q.detach().clone()
        P_prior_safe = P_prior.detach()
        P_target = P_prior if P_target is None else P_target
        P_target = P_target / (P_target.sum(dim=1, keepdim=True) + self.EPS)

        A, _ = model.Attention(
            C,
            Q_safe,
            P_prior_safe,
            detach_P=True,
            detach_Q=True
        )
        dist = self.diag_mahalanobis_distance(C, Q_safe, detach_Q=True, detach_var=True)

        # Train cluster_head / feature path with A as a fixed teacher.
        loss_head = F.kl_div(
            torch.log(P_target.clamp_min(self.EPS)),
            A.detach(),
            reduction='batchmean'
        )

        # Pull C toward the prototype selected by A, without training attention through this path.
        loss_compact = (A.detach() * dist).sum(dim=1).mean()

        # Train the attention projections to assign larger mass to closer prototypes.
        A_attn, _ = model.Attention(
            C.detach(),
            Q_safe,
            P_prior_safe,
            detach_P=True,
            detach_Q=True
        )
        loss_attn = (A_attn * dist.detach()).sum(dim=1).mean()

        loss = 1*loss_head + 0.05 * loss_compact + 0.1 * loss_attn
        return loss
    def compute_attention_A(self, P, Q, C, beta=1.0, eta=1.0, tau=0.2, eps=1e-8, detach_P=True, detach_Q=True):
        """
        Compute prototype-guided sharpened assignment A.

        Args:
            P: Tensor, shape [N, K]
               聚类分配概率矩阵，建议传入 P_tilde.
            Q: Tensor, shape [K, D]
               原型矩阵，建议传入 Q_ema.
            C: Tensor, shape [N, D]
               样本特征矩阵.
            beta:
               控制样本-原型相似度 S 的权重.
            eta:
               控制 P prior 的权重.
            tau:
               softmax 温度，越小 A 越尖锐.
            eps:
               数值稳定项.
            detach_P:
               是否阻断 P 的梯度.
            detach_Q:
               是否阻断 Q 的梯度，若 Q 是 EMA buffer，建议 True.

        Returns:
            A: Tensor, shape [N, K]
               锐化后的样本-原型 assignment matrix.
            S: Tensor, shape [N, K]
               样本与原型的相似度矩阵.
        """

        if detach_P:
            P = P.detach()
        if detach_Q:
            Q = Q.detach()

        # 确保 P 是合法概率分布
        P = P.clamp_min(eps)
        P = P / (P.sum(dim=1, keepdim=True) + eps)

        # S_nk = cosine_similarity(c_n, q_k)
        C_norm = F.normalize(C, dim=1)
        Q_norm = F.normalize(Q, dim=1)
        S = torch.matmul(C_norm, Q_norm.T)  # [N, K]

        # logits_nk = (beta * S_nk + eta * log(P_nk + eps)) / tau
        logits = (beta * S + eta * torch.log(P + eps)) / tau

        A = F.softmax(logits, dim=1)

        return A, S

    def diag_mahalanobis_distance(self, C, Q, eps=1e-8, detach_Q=True, detach_var=True):
        """
        Compute d(C, Q) with diagonal Mahalanobis distance.

        Args:
            C: Tensor, shape [N, D]
               样本特征.
            Q: Tensor, shape [K, D]
               原型特征.
            eps: float
               数值稳定项.
            detach_Q: bool
               若 Q 是 EMA prototype，建议 True.
            detach_var: bool
               是否阻断方差项梯度，建议 True，避免模型通过改变方差钻空子.

        Returns:
            dist: Tensor, shape [N, K]
                  dist[n, k] = d(c_n, q_k)
        """

        if detach_Q:
            Q = Q.detach()

        var_source = C.detach() if detach_var else C
        var = var_source.var(dim=0, unbiased=False).clamp_min(eps)  # [D]

        diff = C.unsqueeze(1) - Q.unsqueeze(0)  # [N, K, D]

        dist = (diff.pow(2) / var.view(1, 1, -1)).sum(dim=-1)  # [N, K]

        return dist

import torch
import torch.nn.functional as F


def target_distribution(q, eps=1e-8):
    """
    DEC / CPMN style target distribution.

    q: [N, K]
    """
    weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(eps)
    p = weight / weight.sum(dim=1, keepdim=True).clamp_min(eps)
    return p


def student_t_assignment(z, prototypes, alpha=1.0, eps=1e-8):
    """
    Compute Student-t soft assignment.

    z: [N, D]
    prototypes: [K, D]

    return:
        q: [N, K]
    """
    dist2 = torch.sum((z.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2), dim=2)

    q = (1.0 + dist2 / alpha).pow(-(alpha + 1.0) / 2.0)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)

    return q


def get_view_prototype(model, view_idx):
    """
    Compatible with CPMN style:
        model.cluster_layer_v0
        model.cluster_layer_v1
        ...
    """
    name = f"cluster_layer_v"
    if not hasattr(model, name):
        raise AttributeError(f"model has no prototype parameter: {name}")

    return getattr(model, name)


def cpmn_student_kl_loss(
    model,
    latent_list,
    mask,
    alpha=1.0,
    confidence_power=1.0,
    min_confidence_weight=0.2,
    eps=1e-8,
):
    """
    CPMN-style Student-t clustering loss for multi-view incomplete data.

    Low-confidence assignments are down-weighted so the DEC target distribution
    does not amplify noisy pseudo-labels before prototypes become stable.
    """

    losses = []

    for v, z in enumerate(latent_list):
        obs_mask = mask[v].squeeze().bool()

        if obs_mask.sum() == 0:
            continue

        z_obs = z[obs_mask]

        # Important: do not use .data here. We want gradients to update prototypes.
        prototypes = get_view_prototype(model, v)

        q = student_t_assignment(
            z=z_obs,
            prototypes=prototypes[v],
            alpha=alpha,
            eps=eps,
        )

        p = target_distribution(q, eps=eps).detach()
        per_sample_kl = F.kl_div(
            torch.log(q.clamp_min(eps)),
            p,
            reduction="none",
        ).sum(dim=1)

        entropy = -(q * torch.log(q.clamp_min(eps))).sum(dim=1)
        max_entropy = torch.log(q.new_tensor(float(q.size(1))))
        confidence = (1.0 - entropy / max_entropy.clamp_min(eps)).clamp(0.0, 1.0).detach()
        if confidence_power != 1.0:
            confidence = confidence.pow(confidence_power)
        sample_weight = confidence.clamp_min(min_confidence_weight)

        loss_v = (per_sample_kl * sample_weight).sum() / sample_weight.sum().clamp_min(eps)
        losses.append(loss_v)

    if len(losses) == 0:
        return torch.tensor(0.0, device=latent_list[0].device)

    return torch.stack(losses).mean()



def plot_similarity_heatmap(A, B, device,
                            normalize=True,
                            title="Similarity Matrix",
                            figsize=(8, 6),
                            cmap="viridis"):
    """
    Compute similarity matrix between two tensors and plot heatmap.

    Args:
        A: torch.Tensor, shape [N, D]
        B: torch.Tensor, shape [M, D]
        device: torch.device
        normalize: whether to use cosine similarity
        title: figure title
        figsize: heatmap size
        cmap: color map

    Returns:
        sim_matrix: torch.Tensor [N, M]
    """

    # move tensors to GPU
    A = A.to(device)
    B = B.to(device)

    # cosine similarity
    if normalize:
        A_norm = torch.nn.functional.normalize(
            A, p=2, dim=1
        )
        B_norm = torch.nn.functional.normalize(
            B, p=2, dim=1
        )

        sim_matrix = torch.mm(
            A_norm,
            B_norm.T
        )

    # dot-product similarity
    else:
        sim_matrix = torch.mm(
            A,
            B.T
        )

    tau = 0.2
    sim_matrix = F.softmax(sim_matrix / tau, dim=1)

    # visualization requires CPU
    sim_np = sim_matrix.detach().cpu().numpy()


    plt.figure(figsize=figsize)

    sns.heatmap(
        sim_np,
        cmap=cmap,
        square=False,
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )

    # plt.title(title)
    # plt.xlabel("B samples")
    # plt.ylabel("A samples")

    plt.tight_layout()
    plt.savefig(
        "similarity_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


    return sim_matrix


# Normalization Prototype
import math
import torch
import torch.nn.functional as F


def mean_normalized_entropy(
    S: torch.Tensor,
    tau: float = 0.1,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    计算跨视图原型相似性矩阵的平均归一化行熵。

    Args:
        S:
            原型相似性矩阵，形状为 [K_source, K_target]。
            S[i, j] 表示源视图原型 i 与目标视图原型 j 的相似度。
        tau:
            Softmax 温度参数。tau 越小，概率分布越尖锐。
        eps:
            防止 log(0) 的数值稳定项。

    Returns:
        mean_entropy:
            所有源原型的平均归一化熵，标量 Tensor，范围为 [0, 1]。
            值越大，表示跨视图原型对应越模糊。
    """
    if not isinstance(S, torch.Tensor):
        S = torch.as_tensor(S, dtype=torch.float32)

    if S.ndim != 2:
        raise ValueError(
            f"S 必须是二维矩阵，但得到的形状为 {tuple(S.shape)}"
        )

    if tau <= 0:
        raise ValueError("tau 必须大于 0。")

    num_target_prototypes = S.shape[1]

    if num_target_prototypes <= 1:
        raise ValueError("目标视图的原型数量必须大于 1。")

    # 将每行相似度转换为对应概率
    probability = F.softmax(S / tau, dim=1)

    # 计算每个源原型的熵
    row_entropy = -torch.sum(
        probability * torch.log(probability.clamp_min(eps)),
        dim=1
    )

    # 使用 log(K) 归一化到 [0, 1]
    normalized_entropy = (
        row_entropy / math.log(num_target_prototypes)
    )

    # 对所有源原型取平均
    mean_entropy = normalized_entropy.mean()

    return mean_entropy

if '__main__' == __name__:
    import ot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x1 = torch.randn((256, 128), device=device)
    x2 = torch.randn((256, 128), device=device)

    print(compute_gromov_wasserstein(x1, x2))


