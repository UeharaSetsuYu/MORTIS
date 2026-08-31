
import torch
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
import os
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None
warnings.filterwarnings("ignore")
from tqdm import tqdm
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.utils.data import random_split
# from units.dataloader import *
import collections

###############################
from model import *
# from network import *
from units.loss import *
from units.IMVC_DATA import *
from units.UMVC_DATA import build_dataset_unpair
from units.NMVC_DATA import build_dataset_nmvc
from units.CMVC_DATA import build_dataset_Multi
from units.unit import *
from units.clustering import *
from units.evaluate import *
from units.Visualization import *
from cluster_quality_metrics import *

import torch.nn.functional as F
from itertools import combinations
from Metric.MTM import MTM
from Metric.RMR import RMR

########### PCC Poof ##########
from Metric.Ablation import *
from Metric.Prototype_Semantic_Consensus.PSC_Test import test_PSC
from Metric.T_Visualization.plot_transition_matrix import plot_transition_matrix
###############################


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_log_file(args, config):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = getattr(args, "dataset", "dataset")
    missing_rate = getattr(args, "missing_rate", "none")
    times = getattr(args, "times", "none")
    seed = getattr(args, "seed", "none")
    log_name = f"{dataset}_mr{missing_rate}_times{times}_seed{seed}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_name)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("============= Training Log =============\n")
        f.write(f"Time: {timestamp}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Missing rate: {missing_rate}\n")
        f.write(f"Times: {times}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Epochs: {config.get('epochs')}\n")
        f.write(f"Pretrain epochs: {config.get('pre_train_epoch')}\n")
        f.write("========================================\n\n")

    return log_path


def write_log(log_path, message):
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(message).rstrip() + "\n")


def log_print(log_path, message):
    print(message)
    write_log(log_path, message)


def Rec_MSE(x_hat_list, view_num, data, criterion, mask):
    loss_list = []
    for view in range(view_num):
        mask_view = mask[view].squeeze().bool()
        if mask_view.sum() == 0:
            continue

        x_hat_view = x_hat_list[view][mask_view]
        data_view = data[view][mask_view]
        mse_loss_vec = criterion.forward_MSE(x_hat_view, data_view)
        loss_list.append(mse_loss_vec)

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=x_hat_list[0].device)

    return torch.stack(loss_list).mean()


def pre_training(model, loader, criterion, view_num, args, device, config, log_path=None):
    optimizer_pre = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_pre, T_max=config['pre_train_epoch'], eta_min=1e-6)
    model.train()
    args.pre_train_epoch = config['pre_train_epoch']
    args.warming_up = config['warming_up']

    pbar = tqdm(
        range(args.pre_train_epoch),
        desc="Pretraining",
        dynamic_ncols=True,
        leave=True,
        bar_format="{l_bar}{bar:24}{r_bar}",
    )
    for epoch in pbar:
        loss_rec = 0.0
        phase = "AE"

        for batch_idx, (xs, mask, idx) in enumerate(loader):
            for view in range(view_num):
                xs[view] = xs[view].to(device)
                mask[view] = mask[view].to(device)



            optimizer_pre.zero_grad()
            latent_list, x_hat_list = model(xs, mask)

            loss_recon = Rec_MSE(x_hat_list, view_num, xs, criterion, mask)
            loss_cluster = torch.zeros((), device=device)
            # Pretraining is reconstruction-only. Pseudo-label supervision is disabled.
            loss_view = loss_recon
            loss_rec += loss_recon.item()


            loss_view.backward()
            optimizer_pre.step()
        scheduler.step()
        batch_count = max(batch_idx + 1, 1)
        avg_rec = loss_rec / batch_count
        avg_total = avg_rec

        pbar.set_postfix_str(
            f"phase={phase:<6} loss={avg_total:9.4f} rec={avg_rec:9.4f} "
        )
        pretrain_msg = (
            f"[Pretrain Epoch {epoch + 1} | {args.pre_train_epoch}] "
            f"Phase: {phase}, "
            f"Loss: {avg_total: .6f}, "
            f"Loss_rec: {avg_rec: .6f} "
        )
        write_log(log_path, pretrain_msg)


def visit_loss_from_aux(aux, eps=1e-8):
    """
    aux: compute_random_walk_T 返回的 aux
         aux[v]["P_q2c"]: [K, N_v]
    """
    loss = 0.0
    valid_views = 0

    for item in aux:
        P_q2c = item["P_q2c"]          # [K, N_v]

        if P_q2c.size(1) == 0:
            continue

        visit = P_q2c.mean(dim=0)     # [N_v]
        visit = visit / (visit.sum() + eps)

        uniform = torch.full_like(
            visit,
            1.0 / visit.numel()
        )

        loss_v = F.kl_div(
            torch.log(visit.clamp_min(eps)),
            uniform,
            reduction="sum"
        )

        loss = loss + loss_v
        valid_views += 1

    return loss / max(valid_views, 1)

def visit_loss(T, eps = 1e-8):

    visit = T.mean(dim=0) / (T.mean(dim=0).sum() + eps)
    uniform = torch.full_like(visit, 1.0 / visit.numel())
    loss = F.kl_div(
        torch.log(visit.clamp_min(eps)),
        uniform,
        reduction="sum"
    )
    return loss


def Transition(x, y, normalize = True,   tau = 0.2, eps = 1e-8):
    if normalize:
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

    sim = x @ y.T              # [N, M]
    P_xy = F.softmax(sim / tau, dim=1)
    P_xy = P_xy / (P_xy.sum(dim=1, keepdim=True) + eps)


    return P_xy


def transition_matrix_power(transition, steps):
    """Compute the Markov matrix power, including P^0 = I."""
    steps = int(steps)
    if steps < 0:
        raise ValueError("prototype_walk_steps must be non-negative")
    return torch.linalg.matrix_power(transition, steps)


def PrototypeMatching(latent_list, mask, epoch, config, model, args, eps=1e-8):
    """
    Cycle-consistency prototype alignment.

    Preferred path uses co-observed paired samples as the bridge between two
    views. When missing_rate=1.0 removes all paired samples, the function falls
    back to an unpaired prototype cycle. The fallback is weaker scientifically,
    but it keeps the probability-transition ideal trainable in fully incomplete
    settings.
    """
    loss_list = []
    view_num = len(latent_list)
    min_pair_samples = int(config.get('prototype_min_pair_samples', 2))
    visit_weight = float(config.get('prototype_visit_weight', 0.05))
    runtime_data_model = str(config.get('runtime_data_model', 'incomplete')).lower()
    runtime_missing_rate = float(config.get('runtime_missing_rate', 0.0))
    has_unpair = runtime_data_model == 'unpair' or 'U' in runtime_data_model.upper()
    full_unpair = has_unpair and runtime_missing_rate >= 1.0 - 1e-8
    pair_assumption = False if has_unpair else bool(config.get('pair_assumption', not full_unpair))
    use_unpaired_fallback = bool(
        config.get('prototype_unpaired_fallback', (not pair_assumption) or runtime_missing_rate >= 1.0 - 1e-8)
    )
    step = int(config.get('prototype_walk_steps', 2))   # fixed 2

    for v in range(view_num):
        mask_v = mask[v].squeeze().bool()
        proto_v = model.cluster_layer_v[v]

        for u in range(v + 1, view_num):
            mask_u = mask[u].squeeze().bool()
            proto_u = model.cluster_layer_v[u]

            pair_mask = mask_v & mask_u
            if pair_assumption and pair_mask.sum() >= min_pair_samples:
                z_v = latent_list[v][pair_mask]
                z_u = latent_list[u][pair_mask]

                A_v = Transition(proto_v, z_u) @ transition_matrix_power(Transition(z_u, z_u), step)  @ Transition(z_u, proto_u)
                B_v =  Transition(proto_u, z_v) @ transition_matrix_power(Transition(z_v, z_v), step)  @ Transition(z_v, proto_v)
                T_v = A_v @ B_v
                loss_v = -torch.log(torch.diagonal(T_v, dim1=-2, dim2=-1).clamp_min(eps)).mean()

                if (epoch + 1) % 100 == 0:
                    print("Cross-view Prototype Reciprocity Matching Rate: ", RMR(A_v, B_v))


                A_u = Transition(proto_u, z_v) @ transition_matrix_power(Transition(z_v, z_v), step) @ Transition(z_v, proto_v)
                B_u = Transition(proto_v, z_u) @ transition_matrix_power(Transition(z_u, z_u), step) @ Transition(z_u, proto_u)
                T_u = A_u @ B_u
                loss_u = -torch.log(torch.diagonal(T_u, dim1=-2, dim2=-1).clamp_min(eps)).mean()

                if epoch == 1990:
                    plot_heatmap_academic(
                        T_u.detach().cpu().numpy(),
                        save_path="transition_heatmap",
                        x_label="Destination Prototype",
                        y_label="Starting Prototype",
                        title=None,
                        cmap="GnBu"  # 冷色调，推荐
                    )
                    plot_heatmap_academic(
                        A_u.detach().cpu().numpy(),
                        save_path="transition_heatmap_A",
                        x_label="Destination Prototype",
                        y_label="Starting Prototype",
                        title=None,
                        cmap="GnBu"  # 冷色调，推荐
                    )

                if (epoch + 1) % 100 == 0:
                    '''
                        PCC: DHA seed-10; 
                        W/O PCC: DHA 
                    '''
                    t_stats = plot_transition_matrix(
                        0.5*(T_u + T_v),
                        dataset=args.dataset,
                        variant=args.PCC_Setting,  # 或 "w-o-PCC", "OT"
                        scenario=args.data_model,
                        rate=args.missing_rate,
                        seed=args.seed,
                        source_view=v,
                        target_view=u,
                        direction=f"return-v{v + 1}-via-v{u + 1}",
                        epoch=epoch + 1,
                    )

                    print(
                        "Mean return probability:",
                        t_stats["mean_return_probability"],
                    )

                pair_loss = 0.5 * (loss_v + loss_u)
                if visit_weight > 0.0:
                    pair_loss = pair_loss + visit_weight * 0.5 * (visit_loss(T_v, eps) + visit_loss(T_u, eps))
                loss_list.append(pair_loss)

            elif use_unpaired_fallback:
                z_v = latent_list[v][mask_v]
                z_u = latent_list[u][mask_u]
                if z_v.size(0) == 0 or z_u.size(0) == 0:
                    continue
                T_v = (
                        Transition(proto_v, z_u)
                        @ transition_matrix_power(Transition(z_u, z_u), step)
                        @ Transition(z_u, proto_u)
                        @ Transition(proto_u, z_v)
                        @ transition_matrix_power(Transition(z_v, z_v), step)
                        @ Transition(z_v, proto_v)
                )
                loss_v = -torch.log(torch.diagonal(T_v, dim1=-2, dim2=-1).clamp_min(eps)).mean()

                T_u = (
                        Transition(proto_u, z_v)
                        @ transition_matrix_power(Transition(z_v, z_v), step)
                        @ Transition(z_v, proto_v)
                        @ Transition(proto_v, z_u)
                        @ transition_matrix_power(Transition(z_u, z_u), step)
                        @ Transition(z_u, proto_u)
                )
                loss_u = -torch.log(torch.diagonal(T_u, dim1=-2, dim2=-1).clamp_min(eps)).mean()

                fallback_loss = 0.5 * (loss_v + loss_u)
                if visit_weight > 0.0:
                    fallback_loss = fallback_loss  + visit_weight * 0.5 * (visit_loss(T_v, eps) + visit_loss(T_u, eps)) * 0


                loss_list.append(fallback_loss)

            if (epoch + 1) % 100 == 0:
                # S = plot_similarity_heatmap(proto_v, proto_u, 'cuda', title = 'Prototype Similarity')
                # # print("Symmetry: ", torch.norm(S - S.T, p = 'fro') ** 2)
                # H_vu = mean_normalized_entropy(S, tau=0.1)
                # H_uv = mean_normalized_entropy(S.T, tau = 0.1)
                # MTM = 1 - 0.5 * (H_vu.item() + H_uv.item())
                print("Cross-view Prototype Matching Trustworthiness Measure: ", MTM(proto_v, proto_u, 'cuda'))

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=latent_list[0].device)

    return torch.stack(loss_list).mean()




def mknn_contrastive_loss(
    latent_list,
    epoch,
    config,
    model,
    mask=None,
    topk=5,
    temperature=0.2,
    cross_weight=1.0,
    intra_weight=1.0,
    use_cross_fallback=False,
    eps=1e-8,

):
    """
    Multi-view MKNN contrastive loss, generalized from CPMN.

    Missing-view latent vectors are filtered by mask before building the MKNN
    graph. Cross-view positives prefer co-observed paired samples, so the loss
    uses actual multi-view correspondence instead of only distributional KNN.
    """

    device = latent_list[0].device
    if mask is None:
        observed_list = [
            torch.ones(z.size(0), dtype=torch.bool, device=z.device)
            for z in latent_list
        ]
    else:
        observed_list = [
            m.squeeze().bool().to(z.device)
            for m, z in zip(mask, latent_list)
        ]

    latent_observed = [
        z[observed]
        for z, observed in zip(latent_list, observed_list)
    ]

    total_loss = torch.tensor(0.0, device=device)
    loss_count = 0

    def safe_topk(k, n):
        return max(1, min(k, n))

    def contrastive_from_pos_mask(sim, pos_mask, self_mask=None):
        """
        sim: [N, M], cosine similarity matrix.
        pos_mask: [N, M], binary positive mask.
        self_mask: optional [N, M] boolean mask for entries removed from denominator.
        """
        valid = pos_mask.sum(dim=1) > 0
        if not valid.any():
            return None

        logits = sim / temperature
        if self_mask is not None:
            logits = logits.masked_fill(self_mask, -1e9)

        log_prob = F.log_softmax(logits, dim=1)
        pos_count = pos_mask.sum(dim=1).clamp_min(eps)
        loss_per_sample = -(pos_mask * log_prob).sum(dim=1) / pos_count

        return loss_per_sample[valid].mean()

    def intra_view_loss(z):
        """
        Intra-view MKNN contrastive loss.
        """
        n = z.size(0)
        if n <= 1:
            return None

        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.t())

        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        sim_no_self = sim.masked_fill(eye, -1e9)

        k = safe_topk(topk * 2, n - 1)
        _, topk_idx = torch.topk(sim_no_self, k=k, dim=1)

        adj = torch.zeros_like(sim, dtype=torch.bool)
        adj.scatter_(1, topk_idx, True)

        mutual = adj & adj.t()

        no_pos = mutual.sum(dim=1) == 0
        if no_pos.any():
            mutual[no_pos] = adj[no_pos]

        pos_mask = mutual.float()
        return contrastive_from_pos_mask(sim, pos_mask, self_mask=eye)

    def cross_view_loss(za, zb, paired=False):
        """
        Cross-view MKNN contrastive loss.

        If paired=True, za[n] and zb[n] are two views of the same sample. The
        diagonal is always a positive pair, and mutual cross-view neighbors are
        added as semantic positives.
        """
        na, nb = za.size(0), zb.size(0)
        if na == 0 or nb == 0:
            return None

        za = F.normalize(za, dim=1)
        zb = F.normalize(zb, dim=1)
        sim = torch.mm(za, zb.t())

        ka = safe_topk(topk * 2, nb)
        kb = safe_topk(topk * 2, na)

        _, topk_a2b = torch.topk(sim, k=ka, dim=1)
        _, topk_b2a = torch.topk(sim, k=kb, dim=0)

        adj_a2b = torch.zeros_like(sim, dtype=torch.bool)
        adj_a2b.scatter_(1, topk_a2b, True)

        adj_b2a = torch.zeros_like(sim, dtype=torch.bool)
        adj_b2a.scatter_(0, topk_b2a, True)

        mutual = adj_a2b & adj_b2a

        if paired and na == nb:
            eye = torch.eye(na, device=sim.device, dtype=torch.bool)
            mutual = mutual | eye
        elif use_cross_fallback:
            no_pos = mutual.sum(dim=1) == 0
            if no_pos.any():
                mutual[no_pos] = adj_a2b[no_pos]

        pos_mask = mutual.float()
        return contrastive_from_pos_mask(sim, pos_mask)

    # 1. Intra-view MKNN contrastive loss
    intra_losses = []
    for z in latent_observed:
        loss_v = intra_view_loss(z)
        if loss_v is not None:
            intra_losses.append(loss_v)

    if len(intra_losses) > 0:
        total_loss = total_loss + intra_weight * torch.stack(intra_losses).mean()
        loss_count += intra_weight

    # 2. Cross-view MKNN contrastive loss. Any mode containing U must not
    # trust row-wise correspondence and therefore uses the unpaired branch.
    runtime_data_model = str(config.get('runtime_data_model', '')).lower()
    runtime_missing_rate = float(config.get('runtime_missing_rate', 0.0))
    has_unpair = runtime_data_model == 'unpair' or 'U' in runtime_data_model.upper()
    pair_assumption = False if has_unpair else bool(config.get('pair_assumption', True))
    use_unpaired_cross = bool(
        use_cross_fallback or has_unpair or runtime_missing_rate >= 1.0 - 1e-8
    )
    cross_losses = []
    for i, j in combinations(range(len(latent_list)), 2):
        pair_mask = observed_list[i] & observed_list[j]
        pair_losses = []

        if pair_assumption and pair_mask.sum() > 1:
            zi = latent_list[i][pair_mask]
            zj = latent_list[j][pair_mask]
            loss_ij = cross_view_loss(zi, zj, paired=True)
            loss_ji = cross_view_loss(zj, zi, paired=True)
        elif use_unpaired_cross:
            loss_ij = cross_view_loss(latent_observed[i], latent_observed[j], paired=False)
            loss_ji = cross_view_loss(latent_observed[j], latent_observed[i], paired=False)
        else:
            loss_ij = None
            loss_ji = None

        if loss_ij is not None:
            pair_losses.append(loss_ij)
        if loss_ji is not None:
            pair_losses.append(loss_ji)

        if len(pair_losses) > 0:
            cross_losses.append(torch.stack(pair_losses).mean())

    if len(cross_losses) > 0:
        total_loss = total_loss + cross_weight * torch.stack(cross_losses).mean()
        loss_count += cross_weight

    if loss_count == 0:
        return torch.tensor(0.0, device=device)

    prototype_update_interval = int(config.get('prototype_update_interval', 10))
    prototype_update_until = int(config.get('prototype_update_until', 100))
    prototype_update_momentum = float(config.get('prototype_update_momentum', 0.9))

    if (
        prototype_update_interval > 0
        and (epoch + 1) % prototype_update_interval == 0
        and (epoch + 1) <= prototype_update_until
    ):
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception:
            linear_sum_assignment = None

        for v, z_obs in enumerate(latent_observed):
            if z_obs.size(0) < config['class_num']:
                continue

            kmeans = KMeans(n_clusters=config['class_num'], random_state=42, n_init=10)
            kmeans.fit(z_obs.detach().cpu().numpy())
            centers = torch.as_tensor(
                kmeans.cluster_centers_,
                dtype=model.cluster_layer_v[v].dtype,
                device=z_obs.device,
            )

            proto = model.cluster_layer_v[v]
            with torch.no_grad():
                if linear_sum_assignment is not None:
                    cost = torch.cdist(proto.detach(), centers).cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost)
                    aligned_centers = proto.detach().clone()
                    for row, col in zip(row_ind, col_ind):
                        aligned_centers[row] = centers[col]
                else:
                    nearest = torch.cdist(proto.detach(), centers).argmin(dim=1)
                    aligned_centers = centers[nearest]

                proto.copy_(
                    proto.detach() * prototype_update_momentum
                    + aligned_centers * (1.0 - prototype_update_momentum)
                )

    return total_loss / loss_count


def Training(args, config):
    setup_seed(args.seed)
    epochs = config['epochs']
    view_num = config['view_num']
    lr = config['learning_rate']
    args.batch_size = config['batch_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_model == 'incomplete':
        dataset = build_dataset(args)  # incomplete dataset
    elif args.data_model == 'unpair':
        dataset = build_dataset_unpair(args)
    elif args.data_model == 'noisy':
        dataset = build_dataset_nmvc(args)
    elif args.data_model in ['UI', 'UN', 'IN', 'UIN']:  #
        dataset = build_dataset_Multi(args)
    else:
        raise ValueError(f"Unsupported data_model: {args.data_model}")


    dims = data_info[args.dataset]['n_input']
    view = data_info[args.dataset]['V']
    data_size = data_info[args.dataset]['N']
    class_num = data_info[args.dataset]['K']



    result = collections.defaultdict(list)
    Loss_list = collections.defaultdict(list)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    dim = [config['Autoencoder']['arch' + str(view + 1)] for view in range(view_num)]
    criterion = Loss()


    model = CausalMVC(config, dim, device)
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.000)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['epochs'] , eta_min = 1e-6)
    log_path = init_log_file(args, config)


    '''Foundation of Training'''
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log_print(log_path, f"Training log saved to: {log_path}")
    log_print(log_path, f"DataName: {args.dataset}")
    log_print(log_path, f"View: {view}, dims of data: {dims}, data size: {data_size}, class num: {class_num}")
    log_print(log_path, device_name)
    log_print(log_path, f"Tringing Scenario: {args.data_model}")
    log_print(log_path, "Training...")


    '''Pretraining'''
    log_print(log_path, "============= Pretraining ==============")
    pre_training(model, data_loader, criterion, view_num, args, device, config, log_path)

    flag = True
    config['runtime_data_model'] = str(getattr(args, 'data_model', 'incomplete')).lower()
    config['runtime_missing_rate'] = float(args.missing_rate)
    has_unpair = config['runtime_data_model'] == 'unpair' or 'U' in config['runtime_data_model'].upper()
    full_unpair = has_unpair and config['runtime_missing_rate'] >= 1.0 - 1e-8
    config['pair_assumption'] = (
        False if has_unpair else bool(config.get('pair_assumption', not full_unpair))
    )
    no_pair_mode = not config['pair_assumption']
    schedule_base = int(config.get('warming_up', 100))
    proto_start_epoch = int(config.get('prototype_start_epoch', max(10, int(schedule_base * 0.2))))
    proto_warmup_epoch = int(config.get('prototype_warmup_epoch', max(20, int(schedule_base * 0.5))))
    min_mknn_weight = float(config.get('min_mknn_weight', 0.2))
    unpaired_match_scale = float(config.get('unpaired_match_weight_scale', 0.3 if no_pair_mode else 1.0))
    effective_match_scale = unpaired_match_scale if no_pair_mode else 1.0
    paired_cross_weight = float(config.get('mknn_paired_cross_weight', 1.0))
    unpaired_cross_weight = float(config.get('mknn_unpaired_cross_weight', 0.1))
    mknn_cross_weight = float(config.get(
        'mknn_cross_weight',
        unpaired_cross_weight if has_unpair else paired_cross_weight,
    ))
    mknn_topk = int(config.get(
        'mknn_topk',
        config.get('mknn_unpaired_topk', 1) if has_unpair else config.get('mknn_paired_topk', 3),
    ))
    mknn_temperature = float(config.get('mknn_temperature', 0.2))
    mknn_intra_weight = float(config.get('mknn_intra_weight', 1.0))
    mknn_unpaired_fallback = bool(config.get('mknn_unpaired_fallback', False))
    log_print(
        log_path,
        f"Prototype schedule: start={proto_start_epoch}, warmup={proto_warmup_epoch}, "
        f"min_mknn_weight={min_mknn_weight:.3f}, "
        f"visit_weight={float(config.get('prototype_visit_weight', 0.05)):.3f}, "
        f"clu_min_conf={float(config.get('clu_min_confidence_weight', 0.2)):.3f}, "
        f"data_model={config['runtime_data_model']}, missing_rate={config['runtime_missing_rate']:.3f}, "
        f"pair_assumption={config['pair_assumption']}, no_pair_mode={no_pair_mode}, "
        f"effective_match_scale={effective_match_scale:.3f}, "
        f"mknn_topk={mknn_topk}, mknn_intra_weight={mknn_intra_weight:.3f}, "
        f"mknn_cross_weight={mknn_cross_weight:.3f}, "
        f"mknn_unpaired_fallback={mknn_unpaired_fallback}, "
        f"prototype_walk_steps={int(config.get('prototype_walk_steps', 0))}"
    )

    for epoch in range(epochs):
        model.train()
        loss_all, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6 = 0, 0, 0, 0, 0, 0, 0

        if (epoch + 1) >= proto_start_epoch:
            proto_progress = min(1.0, (epoch + 1 - proto_start_epoch + 1) / max(proto_warmup_epoch, 1))
        else:
            proto_progress = 0.0
        mknn_weight = args.lamda_1 * max(min_mknn_weight, 1.0 - 0.5 * proto_progress)
        match_weight = args.lamda_2 * proto_progress * effective_match_scale
        clu_weight = args.lamda_3 * proto_progress

        for batch_idx, (xs, mask, idx) in enumerate(data_loader):
            # print(xs[0].shape)
            for view in range(view_num):
                xs[view] = xs[view].to(device)
                mask[view] = mask[view].to(device)
            if flag == True:
                if profile is not None and clever_format is not None:
                    flops, params = profile(model, inputs=(xs, mask))
                    flops_str, params_str = clever_format([flops, params], "%.3f")
                    tqdm.write(f"FLOPs: {flops_str}, Params: {params_str}")
                    write_log(log_path, f"FLOPs: {flops_str}, Params: {params_str}")
                else:
                    write_log(log_path, "FLOPs: skipped because thop is not installed")
                flag = False

            latent_list, x_hat_list  = model(xs, mask)


            loss_rec = Rec_MSE(x_hat_list, view_num, xs, criterion, mask)
            loss = loss_rec
            loss_1 += loss_rec.item()


            # loss_walker, loss_walk, loss_visit, loss_cvt, loss_pro, loss_orth = PrototypeLearning(con_latent_list, latent_list, mask, Q, criterion, A, epoch)
            # loss_2 += loss_walk
            # loss_3 += loss_visit
            # loss_4 += loss_cvt
            # loss_5 += loss_pro
            # loss_6 += loss_orth
            loss_mknn = mknn_contrastive_loss(
                latent_list=latent_list,
                mask = mask,
                topk=mknn_topk,
                temperature=mknn_temperature,
                intra_weight=mknn_intra_weight,
                cross_weight=mknn_cross_weight,
                use_cross_fallback=mknn_unpaired_fallback,
                epoch = epoch,
                model = model, config = config,
            )
            loss_2 += loss_mknn.item()
            loss += mknn_weight * loss_mknn

            if proto_progress > 0.0:

                loss_trans = PrototypeMatching(latent_list, mask, epoch, config, model, args) # PCC
                # loss_trans = Directly_ProtoAlign(latent_list, model, epoch) # MSE-based
                # loss_trans = Robust_Loss(latent_list, model, epoch) # Robust Contrastive Loss
                # loss_trans = OT_align(latent_list, model, epoch)
                #


                loss_3 +=  loss_trans.item()
                loss += match_weight * loss_trans

                loss_clu = cpmn_student_kl_loss(
                    model,
                    latent_list,
                    mask,
                    alpha=1.0,
                    confidence_power=float(config.get('clu_confidence_power', 1.0)),
                    min_confidence_weight=float(config.get('clu_min_confidence_weight', 0.2)),
                )
                loss_4 += loss_clu.item()
                loss += clu_weight * loss_clu

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()



        if args.dataset != 'DHA' or '  ' or ' ':
            scheduler.step()  #

        epoch_msg = (
            f'[Epoch {epoch + 1} | {epochs}] Loss: {loss_all: .6f}, Loss_rec: {loss_1: .6f}, '
            f'Loss_MKnn: {loss_2: .6f}, Loss_Match: {loss_3: .6f}, '
            # f'loss_Cvt: {loss_4: .6f}, Loss_Pro: {loss_5: .6f}, '
            f'Loss_clu: {loss_4: .6f}, '
            f'w_mknn: {mknn_weight:.4f}, w_match: {match_weight:.4f}, w_clu: {clu_weight:.4f}'
        )
        log_print(log_path, epoch_msg)

        Loss_list['Loss_All'].append(loss_all)
        # Loss_list['Loss_Walk'].append(loss_temp2)
        Loss_list['Loss_Comp'].append(loss_2)

        if (epoch + 1) % args.test_times == 0:
            acc, ari, nmi, pur = test_Kmeans(model, dataset, view_num, args, device, epoch, log_path=log_path)

            test_msg = f'Final Test -> acc: {acc:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, pur: {pur:.4f}\n'
            log_print(log_path, test_msg)

            result['ACC'].append(acc)
            result['ARI'].append(ari)
            result['NMI'].append(nmi)
            result['PUR'].append(pur)

            psc_result = test_PSC(
                model=model,
                dataset=dataset,
                view_num=view_num,
                args=args,
                device=device,
                epoch=epoch,
                log_path=log_path,
            )
            psc = psc_result["PSC"]
            purity = psc_result["Purity_mean"]
            hard_purity = psc_result["Hard_Purity_mean"]
            print(f"PSC: {psc}, Purity: {purity}, Hard-Purity: {hard_purity}")

    return result, Loss_list



# Clustering by KMeans
def test_Kmeans(model, dataset, view_num, args, device, epoch, log_path=None):
    model.eval()

    # 测试的时候，就不要把大家打乱，也不要无情地落下任何人了哦
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 保持大家原本的顺序
        drop_last=False,  # 就算最后不够一个 Batch，也要带上他们
    )
    labels_ = data_loader.dataset.labels.squeeze()
    all_latent_fusion = []
    raw_data = []
    for batch_idx, (xs, mask, idx) in enumerate(data_loader):
        for view in range(view_num):
            xs[view] = xs[view].to(device)
            mask[view] = mask[view].to(device)
        with torch.no_grad():
            latent_list, _ = model(xs, mask)
            eval_mode = str(getattr(args, 'data_model', 'incomplete')).upper()
            if eval_mode == 'UNPAIR' or 'U' in eval_mode:
                anchor_view = int(getattr(args, 'anchor_view', 0))
                z = latent_list[anchor_view]
            else:
                mask = [mask[i].squeeze() for i in range(view_num)]
                mask = torch.stack(mask)
                C_list = [latent_list[view] * mask[view].unsqueeze(1) for view in range(view_num)]
                observed_count = mask.sum(dim=0).clamp_min(1.0).unsqueeze(1)
                z = torch.stack(C_list, dim=0).sum(dim=0) / observed_count

            latent_fusion = z.cpu().numpy()
            all_latent_fusion.append(latent_fusion)
            raw_data.append(xs[0].cpu().numpy())


    Z_global = np.concatenate(all_latent_fusion, axis = 0)
    raw_data_global = np.concatenate(raw_data, axis = 0)
    y_pre_global = kmeans_with_alignment(Z_global, labels_)
    acc, ari, nmi, pur = evaluate(labels_, y_pre_global)


    '''Prototype Evaluation'''
    Z = torch.tensor(Z_global, device = device)
    rawData = torch.tensor(raw_data_global, device = device)


    db_index = compute_db_index(Z, y_pre_global)
    sc_score = compute_sc_score(Z, y_pre_global)
    log_print(log_path, f"Cluster Quality: DB-{db_index}, SC-{sc_score}")



    if (epoch + 1) % 100 == 0  or (epoch + 1) == 1:
        # fig = plot_tsne(Z, labels_, Norm=False, savefilename='true_label_clustering_tsne.png', dataname = args.dataset)
        # fig = plot_tsne(Z, y_pre_global, Norm=False, savefilename='label_pre_clustering_tsne.png')
        if args.dataset == 'Caltech5V':
            fig, embedding_2d = plot_tsne(
                features=Z_global,
                true_labels=labels_,
                title=None,
                random_state=42,
                perplexity=50.0,    # 30
                pca_dim=21, # 50
                savefilename=f"clustering_tsne_epoch_{epoch + 1}.png",
                dataname=args.dataset,
            )
        elif args.dataset == 'CUB':
            fig, embedding_2d = plot_tsne(
                features=Z_global,
                true_labels=labels_,
                title=None,
                random_state=42,
                perplexity=20.0,
                pca_dim=30,
                savefilename=f"clustering_tsne_epoch_{epoch + 1}.png",
                dataname=args.dataset,
            )
        else:
            fig, embedding_2d = plot_tsne(
                features=Z_global,
                true_labels=labels_,
                title=None,
                random_state=42,
                perplexity=30.0,
                pca_dim=50,
                savefilename=f"clustering_tsne_epoch_{epoch + 1}.png",
                dataname=args.dataset,
            )




    return acc, ari, nmi, pur
def Test_cluster(model, dataset, view_num, args, device, epoch):
    model.eval()

    # 测试的时候，就不要把大家打乱，也不要无情地落下任何人了哦
    # （大家一定要一直在一起呢...）
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 保持大家原本的顺序
        drop_last=False,  # 就算最后不够一个 Batch，也要带上他们
    )
    labels_ = data_loader.dataset.labels.squeeze()
    all_latent_fusion = []
    raw_data = []
    Z = []
    for batch_idx, (xs, mask, idx) in enumerate(data_loader):
        for view in range(view_num):
            xs[view] = xs[view].to(device)
            mask[view] = mask[view].to(device)

        with torch.no_grad():
            _, _, C, P_list, *_ = model(xs, mask)
            mask = [mask[i].squeeze() for i in range(view_num)]
            mask = torch.stack(mask)

            C_list = [C[view] * mask[view].unsqueeze(1) for view in range(view_num)]

            z = torch.stack(C_list, dim = 0).sum(dim = 0) / mask.sum(dim = 0).unsqueeze(1)
            # z = C

            z = z.cpu().numpy()
            Z.append(z)

            # P_tensor 的形状会是: [view_num, batch_size, num_clusters]
            P_tensor = torch.stack([P_list[view] * mask[view].unsqueeze(1) for view in range(view_num)])

            # 1. 跨视图比较：找出每个样本分到各个簇的“最大概率”
            # max_prob_across_views 的形状是: [batch_size, num_clusters]
            max_prob_across_views, _ = P_tensor.max(dim=0)

            # 2. 簇内比较：在这个最大概率中，选出概率最大的那个簇作为大家最终的归宿
            # final_cluster_idx 的形状是: [batch_size]
            _, final_cluster_idx = max_prob_across_views.max(dim=1)

            latent_fusion = final_cluster_idx.cpu().numpy()
            all_latent_fusion.append(latent_fusion)
            raw_data.append(xs[0].cpu().numpy())

    Z_global = np.concatenate(all_latent_fusion, axis=0)
    raw_data_global = np.concatenate(raw_data, axis=0)
    Z = np.concatenate(Z, axis=0)
    y_pre_global = Z_global
    # print(y_pre_global)

    # 看看大家是不是真的被好好分在一起了...
    acc, ari, nmi, pur = evaluate(labels_, y_pre_global)

    if (epoch + 1) == 5 or (epoch + 1) % 100 == 0:
        fig = plot_tsne(Z, labels_, Norm=False, savefilename='true_label_clustering_tsne.png')


    return acc, ari, nmi, pur


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_tsne_with_prototypes(C, P):
    """
    将GPU上的张量 C 和 P 进行 t-SNE 降维并可视化。
    C: 样本矩阵，形状为 (N, D)
    P: 原型矩阵，形状为 (M, D)
    """
    # 那个... 必须先从GPU移到CPU上，并转换成numpy格式呢
    C_np = C.detach().cpu().numpy()
    P_np = P.detach().cpu().numpy()

    # 把大家聚在一起... 只有一起做t-SNE，才能在一个空间里哦
    combined = np.vstack((C_np, P_np))

    # 初始化 t-SNE
    # 如果数据量很大，可能需要调整 perplexity 参数
    tsne = TSNE(n_components=2, random_state=42)

    # 降维
    combined_tsne = tsne.fit_transform(combined)

    # 降维后再把大家分开
    num_c = C_np.shape[0]
    C_tsne = combined_tsne[:num_c, :]
    P_tsne = combined_tsne[num_c:, :]

    # 开始画图
    plt.figure(figsize=(10, 8))

    # 画样本 C (用普通的圆点表示)
    plt.scatter(C_tsne[:, 0], C_tsne[:, 1], c='#1f77b4', label='Samples (C)', alpha=0.6, s=30)

    # 画原型 P (按照你的要求，用特别的星号标记，稍微画大一点会更清楚呢)
    plt.scatter(P_tsne[:, 0], P_tsne[:, 1], c='red', label='Prototypes (P)',
                marker='*', s=250, edgecolors='black', linewidths=1)

    plt.title('t-SNE Visualization of Samples and Prototypes')
    plt.legend()
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 显示出来
    plt.show()
























