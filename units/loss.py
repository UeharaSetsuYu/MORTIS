import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from units.unit import *





class Noise_robust_loss(nn.Module):
    def __init__(self):
        super(Noise_robust_loss, self).__init__()
    def forward(self,h0, h1):
        h0, h1 = nn.functional.normalize(h0, dim=1), nn.functional.normalize(h1, dim=1)
        tao = 1
        t = 3
        cos = self.get_Similarity(h0, h1)
        sim = (cos/tao).exp()
        pos = sim.diag()
        Q = pos / sim.sum(1)
        result = 0.0000
        for i in range(1, t):
            result += (((1 - Q) ** i) / i).mean()
        robust_loss = result
        return robust_loss

    def get_Similarity(self, fea_mat1, fea_mat2):
        Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
        return Sim_mat




class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.MSE = nn.MSELoss() # reduction = 'none'
        self.Wasserstein = SinkhornDistance(eps = 0.1)
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.NoiseContrastiveLoss = Noise_robust_loss()
        self.CompactLoss = ProtoFeatureLoss()
    def forward(self, input, target):
        return []

    def forward_MSE(self, x, x_hat):
        return self.MSE(x, x_hat)

    def forward_MutualInformation(self, z1, z2, temperature=0.1):
        batch_size = z1.size(0)
        device = z1.device

        # 为了公平起见，先把大家都在特征维度上做 L2 归一化
        # 这样点乘的结果就直接是余弦相似度了呢
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        # 计算所有样本两两之间的相似度矩阵 (Batch_Size x Batch_Size)
        # 对角线上的就是原本属于同一个样本的 z1 和 z2 (正样本对)
        sim_matrix = torch.matmul(z1_norm, z2_norm.T) / temperature

        # 制造标签：因为正样本对都在对角线上，所以标签就是 0 到 Batch_Size - 1
        labels = torch.arange(batch_size, device=device)

        # 用交叉熵来计算 InfoNCE 损失
        loss_infonce = F.cross_entropy(sim_matrix, labels)

        # 互信息的下界估计 = log(Batch_Size) - InfoNCE_Loss
        # 这个值越大，说明 z1 和 z2 包含的共享信息越多哦
        mi_estimate = torch.log(torch.tensor(batch_size, dtype=torch.float32, device=device)) - loss_infonce

        return mi_estimate


    def forward_BCE(self, discriminator_output, target_labels):
        return self.BCE(discriminator_output, target_labels)

    def forward_Wasserstein(self, X, Y):
        return self.Wasserstein(X, Y)



    def forward_KL(self, x, y, reduction = 'batchmean'):
        # y 是目标分布， x是需要预测的分布
        return F.kl_div(torch.log(x), y, reduction = reduction)

    def forward_CrossEntropy(self, label_pred, label_true):
        return F.cross_entropy(label_pred, label_true)

    def forward_Crossentropy_(self, x, y):
        similarity = -torch.log(torch.softmax(y, dim=1))

        nll_loss = similarity * x / x.sum(dim=1, keepdim=True)

        loss = nll_loss.mean()
        return loss

    def forward_KL_symmetry(self, label_pred, label_true, step = 2):
        loss = (F.cross_entropy(label_pred, label_true) + F.cross_entropy(label_true, label_pred)) / step

        return loss









    def forward_FNorm(self, x, y):
        f_norm = torch.linalg.norm(x - y, ord='fro')
        return f_norm

    def forward_orthogonal_loss(self, shared, specific, tau=0.01):
        """Penalize correlation between shared and view-specific features"""
        feature_dim = specific.shape[1]
        # Center shared features
        _shared = shared - shared.mean(dim=0)
        # Compute correlation matrix
        correlation_matrix = _shared.t().matmul(specific)
        # Measure deviation from orthogonal (diagonal should match feature dimension)
        trace_diff = torch.abs(correlation_matrix.trace() - feature_dim) * tau
        return trace_diff

    def structural_alignment_loss(self, M1, M2, temperature=0.05):
        """
        完全按照 Lin 的直觉实现的结构对比损失

        参数:
        M1, M2 (torch.Tensor): 两个视图的转移概率矩阵或结构表征，形状为 [B, D]
        temperature (float): 温度系数，用来放大相似度的差异
        """
        batch_size = M1.shape[0]

        # 1. 保护性操作：L2 归一化
        # 防止因为 M 里的数值太小，导致相似度矩阵坍缩
        M1_norm = F.normalize(M1, p=2, dim=1)
        M2_norm = F.normalize(M2, p=2, dim=1)

        # 2. 计算相似性矩阵 sim(M1, M2)
        # 除以温度系数，让稍后的 Softmax 分布更加尖锐
        sim_matrix = torch.matmul(M1_norm, M2_norm.T) / temperature

        # 3. 生成标签：我们希望对角线上的值最大
        # 也就是说，第 i 行的目标类别就是 i
        labels = torch.arange(batch_size).to(M1.device)

        # 4. Softmax + 交叉熵计算
        # PyTorch 的 cross_entropy 会自动对 sim_matrix 每行做 Softmax 并算 Loss
        # 计算 M1 找 M2 (视图 1 匹配视图 2)
        loss_12 = F.cross_entropy(sim_matrix, labels)

        # 计算 M2 找 M1 (视图 2 匹配视图 1) -> 对称损失通常会让对齐更稳定哦
        loss_21 = F.cross_entropy(sim_matrix.T, labels)

        # 返回平均的对比损失
        return (loss_12 + loss_21) / 2.0

    def contrastive_loss(self, z_i, z_j, temperature=0.5, tau=0.01):
        """Compute contrastive loss between positive pairs and negative samples"""
        batch_size = z_i.shape[0]
        N = 2 * batch_size  # Total number of samples (original + augmented)
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)  # Mask self-comparisons
        # Remove positive pair connections
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()  # Convert to boolean mask

        # Concatenate original and augmented samples
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix between all samples
        sim_matrix = torch.matmul(z, z.T) / temperature
        # Get positive pair similarities (diagonal elements)
        sim_i_j = torch.diag(sim_matrix, batch_size)  # Original vs augmented
        sim_j_i = torch.diag(sim_matrix, -batch_size)  # Augmented vs original

        # Combine positive similarities
        pos_sim = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # Extract negative similarities using mask
        neg_sim = sim_matrix[mask].reshape(N, -1)

        # Cross-entropy loss calculation
        labels = torch.zeros(N).to(pos_sim.device).long()  # All positives are at index 0
        logits = torch.cat((pos_sim, neg_sim), dim=1)  # Combine positive and negatives
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = criterion(logits, labels)

        return loss / N * tau  # Normalized loss with temperature scaling

    def get_sim(self,centroids):
        """Calculate similarity structure between centroids.

        Args:
            centroids: Input centroid matrix [class_num, feature_dim]

        Returns:
            Softmax-normalized similarity matrix [class_num, class_num]
        """
        # Calculate pairwise similarity scores
        structure = torch.mm(centroids, centroids.T)

        # Convert to probability distribution per row
        return F.softmax(structure, dim=1)

    def cas_loss(self, cons_centroids, spec_centroids):
        """Cross-view structure alignment loss"""
        # Get similarity structures for both spaces
        cons_structure_sim = self.get_sim(cons_centroids)  # Common space similarity
        spec_structure_sim = self.get_sim(spec_centroids)  # Specific space similarity
        # Compare similarity structures using contrastive loss

        # loss_csa = self.contrastive_loss(cons_structure_sim, spec_structure_sim)
        loss_csa = self.NoiseContrastiveLoss(cons_structure_sim, spec_structure_sim)
        return loss_csa

    # 计算 MI(P) = H(P) - H(mean(P))
    def mi_loss(slef, P, eps=1e-8):
        # P = P / (P.sum(dim=1, keepdim=True) + eps)

        sample_entropy = -(P * torch.log(P + eps)).sum(dim=1).mean()

        marginal = P.mean(dim=0)
        marginal_entropy = -(marginal * torch.log(marginal + eps)).sum()

        return sample_entropy - marginal_entropy

    def rwm_loss(self, z1, z2):
        rwm_1 = kernel_affinity(z1, step = 3, alpha = 0.)
        rwm_2 = kernel_affinity(z2, step = 3, alpha = 0.)
        loss_rwm = self.contrastive_loss(rwm_1, rwm_2)

        return loss_rwm

    def NoiseContrastive(self, x, y):
        return self.NoiseContrastiveLoss(x, y)
    def forward_GWLoss(self, x, y):
        return compute_gromov_wasserstein(x, y, epsilon=1e-2)

    def compute_wasserstein_distance(self, X, Y, epsilon=1e-2):
        """
        计算两个视图特征 X 和 Y 之间的 Wasserstein 距离。
        【重要】X 和 Y 的特征维度必须相同！

        参数:
            X: torch.Tensor, shape (n, d), 位于 GPU
            Y: torch.Tensor, shape (m, d), 位于 GPU
            epsilon: float, 熵正则化系数 (推荐开启，使用 Sinkhorn 迭代)

        返回:
            wd_dist: torch.Tensor, 标量, 两个分布之间的 WD 距离
        """
        # 提前做一下维度检查，免得后面报错呢...
        if X.size(1) != Y.size(1):
            raise ValueError(f"那个... X 的维度 ({X.size(1)}) 和 Y 的维度 ({Y.size(1)}) 不一样，标准 WD 是算不了的哦。")

        # 确保精度
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)

        n = X.size(0)
        m = Y.size(0)

        # 1. 计算跨视图的代价矩阵 (Cost matrix)
        # 注意：这里和 GW 不同，是直接计算 X 和 Y 之间的 L2 距离
        C = torch.cdist(X, Y, p=2)

        # 为了数值稳定进行归一化，让梯度的回传更温柔一点
        C = C / (C.max() + 1e-8)

        # 2. 定义边缘分布 (假设每个样本的权重是均匀的)
        p = torch.ones(n, device=X.device, dtype=X.dtype) / n
        q = torch.ones(m, device=Y.device, dtype=Y.dtype) / m

        # 3. 计算 WD 距离
        # 只要传入的是 GPU 上的 Tensor，POT 就会自动切入 PyTorch 后端
        if epsilon > 0:
            # 开启熵正则化，变成 Sinkhorn 算法。
            # 强烈建议在神经网络里使用这个！在 GPU 上极快，而且梯度非常平滑
            wd_dist = ot.sinkhorn2(p, q, C, epsilon)
        else:
            # 如果要求精确解（纯 EMD），有时候在 GPU 上的反向传播会遇到一点点问题...
            # 深度学习里一般不推荐设为 0 呢。
            wd_dist = ot.emd2(p, q, C)

        return wd_dist

    def Guide_Loss(self, C, Z):
        return self.GuideLoss(C, Z)

    def Prototype_view_intra(self, C, P, Proto):
        return sample_prototype_compact_loss(C, P, Proto)

    def Prototype_Assignment(self, C, P, Proto):
        return sample_proto_similarity_loss(C, Proto, P, temperature = 0.2)

    def Tstudent(self, P):
        return calc_tstudent_clustering_loss(P)

    def Compact_Loss(self, P, Q, C, model, P_target=None):
        # P is the attention prior, usually P_tilde. P_target is usually raw P from cluster_head.
        return self.CompactLoss.forward_Attention(model, C, Q, P, P_target=P_target)

    def balance_loss(self, P, eps=1e-8):
        P = P / (P.sum(dim=1, keepdim=True) + eps)
        marginal = P.mean(dim=0)
        K = P.size(1)
        uniform = torch.full_like(marginal, 1.0 / K)

        return torch.sum(
            marginal * torch.log((marginal + eps) / (uniform + eps))
        )

    def PrototypeLoss(self, P, Q, C):
        MaScore = Ma_Proto_Eva()
        loss = MaScore(Q, P, C)
        return loss

    def MMDLoss(self, x, y):
        return mmd_loss(x, y)
def sample_prototype_compact_loss(C_view, P_view, Proto, tau = 0.5):
    C_view = F.normalize(C_view, dim=1)
    Proto = F.normalize(Proto, dim=1)
    dist = torch.cdist(C_view, Proto, p=2) ** 2
    loss = (P_view * dist).sum(dim=1).mean()
    return loss


def calc_tstudent_clustering_loss(P):
    """
    直接根据预测概率矩阵 P 计算 t-Student 聚类损失 (KL 散度)

    参数:
    P (torch.Tensor): 模型的预测簇分配概率矩阵，维度为 [N, K]，
                      通常是经过 t-Student 公式或 Softmax 计算后的结果。

    返回:
    torch.Tensor: 计算好的标量损失值
    """
    # 为了数值稳定性，添加一个极小值
    eps = 1e-8

    # 1. 计算目标分布 Q
    # 统计每个簇的软频率 (按列求和)
    f = torch.sum(P, dim=0)

    # 计算分子：P^2 / f
    weight = (P ** 2) / (f + eps)

    # 按行归一化得到目标分布 Q
    Q = weight / torch.sum(weight, dim=1, keepdim=True)

    # 2. 截断目标分布的梯度 (极其重要：Q 仅作为伪标签，不参与反向传播更新)
    Q = Q.detach()

    # 3. 计算 KL 散度损失: KL(Q || P) = \sum Q * log(Q / P)
    # 这里也可以用 F.kl_div(torch.log(P + eps), Q, reduction='batchmean')
    loss =  F.kl_div(torch.log(P + eps), Q, reduction='batchmean')

    # 取样本平均，保持 Loss 量级稳定
    # loss = loss / P.shape[0]

    return loss


def sample_proto_similarity_loss(C_view, Proto, P_tilde, temperature=0.2):
    C_view = F.normalize(C_view, dim=1)
    Proto = F.normalize(Proto, dim=1)

    logits = torch.matmul(C_view, Proto.T) / temperature
    log_prob = F.log_softmax(logits, dim=1)

    target = P_tilde.detach()
    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)

    return F.kl_div(log_prob, target, reduction='batchmean')



def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    x: [N, d]
    y: [M, d]
    return:
        kernel matrix [N, M]
    """
    n = x.size(0)
    m = y.size(0)

    total = torch.cat([x, y], dim=0)  # [N+M, d]

    total0 = total.unsqueeze(0)       # [1, N+M, d]
    total1 = total.unsqueeze(1)       # [N+M, 1, d]

    l2_distance = ((total0 - total1) ** 2).sum(2)  # [N+M, N+M]

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / ((n + m) ** 2 - (n + m))

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [
        torch.exp(-l2_distance / bw)
        for bw in bandwidth_list
    ]

    return sum(kernel_val)  # [N+M, N+M]


def mmd_loss(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    x: [N, d]
    y: [M, d]

    return:
        scalar MMD loss
    """
    n = x.size(0)
    m = y.size(0)

    kernels = gaussian_kernel(
        x, y,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )

    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()

    return loss

if '__main__' == __name__:
    x1 = torch.randn((256, 128))
    x2 = torch.randn((256, 128))
    print(mahalanobis_info_nce_loss(x1, x2))
