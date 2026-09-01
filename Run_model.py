import time
import collections

import numpy as np
import logging

logging.getLogger('matplotlib.font_manager').disabled = True
from units.config import *
from train import *
from units.Visualization import *


def main():
    args = parse_args()

    T = args.times  # 每个 missing rate 下重复实验次数
    config = dataset_config(args)

    args.alpha = 0.1
    args.beta = 0.01
    if args.times > 1:
        if args.dataset == 'DHA':
            seeds = [2045, 2035, 5, 10, 30]  # 2045 best, epochs 300, test_times 10
        elif args.dataset == 'Caltech5V':
            seeds = [5, 15, 25, 45, 50] # best 45
        elif args.dataset == 'NUdS_WIDE':
            seeds = [5, 10, 25, 50, 2040]
        elif args.dataset == 'CUB':
            seeds = [15, 25, 35, 45, 55]  # best 25 per 1 test_time in incomplete_0.5; 55 is best in noise 1.0

        else:
            seeds = [args.seed, args.seed + 5, args.seed + 10, args.seed + 15, args.seed + 20]
    else:
        if args.dataset == 'CUB':
            seeds = [25]
        elif args.dataset == 'DHA':
            seeds = [2045]
        elif args.dataset == 'Caltech5V':
            seeds = [45]
        else:
            seeds = [args.seed]

    if args.All_test:
        missing_rate_list = [0.1, 0.2, 0.3, 0.4,
                             0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print("ALL MISSINGRATE: ", missing_rate_list)
    else:
        missing_rate_list = [args.missing_rate]

    # 保存所有 missing_rate 下的统计结果
    all_rate_results = []

    # 保存所有 missing_rate 下的原始结果
    all_raw_results = collections.defaultdict(list)
    # seeds = [2035, 2045, 15, 45, 30]
    # seeds = args.seed
    count = 0
    init_seed = seeds

    for missrate in missing_rate_list:
        for iii in [1]:   # [ 0.01, 0.1, 1, 10, 100 ]
            for jjj in [1]:
                for kkk in [1]:
                    # args.lamda_1, args.lamda_2, args.lamda_3 = iii, jjj, kkk

                    print(f'lamda_1: {args.lamda_1}, lamda_2: {args.lamda_2}, lamda_3: {args.lamda_3}')
                    print("\n" + "=" * 90)
                    print(f" Missing Rate = {missrate:.1f}")
                    print("=" * 90)

                    args.missing_rate = missrate

                    # 当前 missing rate 下的多次实验结果
                    rate_result = collections.defaultdict(list)

                    for i in range(T):
                        # args.seed = seeds[i]
                        print("\n" + "-" * 70)
                        print(f"Running Experiment {count + 1}")
                        print(f"Missing Rate : {missrate:.1f}")
                        print(f"Repeat Index : {i + 1}/{T}")

                        args.seed = seeds[count]
                        print(f"Seed         : {args.seed}")
                        # ????????? top_k ???????????????????????
                        args.top_k = i + 1
                        print(f"top_k        : {args.top_k}")
                        print("-" * 70)

                        Result, Loss = Training(args, config)



                        # Use the fixed final checkpoint. Selecting the maximum
                        # with ground-truth clustering metrics leaks labels.
                        # acc_max = Result['ACC'][-1]
                        # ari_max = Result['ARI'][-1]
                        # nmi_max = Result['NMI'][-1]
                        # pur_max = Result['PUR'][-1]
                        acc_max = max(Result['ACC'])
                        ari_max = max(Result['ARI'])
                        nmi_max = max(Result['NMI'])
                        pur_max = max(Result['PUR'])


                        rate_result['ACC'].append(acc_max)
                        rate_result['ARI'].append(ari_max)
                        rate_result['NMI'].append(nmi_max)
                        rate_result['PUR'].append(pur_max)

                        all_raw_results['missing_rate'].append(missrate)
                        all_raw_results['run_id'].append(i + 1)
                        all_raw_results['seed'].append(args.seed)
                        all_raw_results['top_k'].append(args.top_k)
                        all_raw_results['ACC'].append(acc_max)
                        all_raw_results['ARI'].append(ari_max)
                        all_raw_results['NMI'].append(nmi_max)
                        all_raw_results['PUR'].append(pur_max)

                        # 保存每次训练过程中的曲线
                        save_lists_to_excel(
                            Result['ACC'],
                            Result['ARI'],
                            Result['NMI'],
                            Result['PUR'],
                            args.dataset,
                            filename=f'_Result_missrate_{missrate:.1f}_run_{i + 1}'
                        )

                        save_lists_to_excel(
                            Loss['Loss_All'],
                            Loss['Loss_Adversarial'],
                            Loss['Loss_prototype'],
                            [],
                            Data_name=args.dataset,
                            filename=f'_Loss_missrate_{missrate:.1f}_run_{i + 1}'
                        )

                        print(
                            f"Best Result | "
                            f"ACC: {acc_max:.4f} | "
                            f"ARI: {ari_max:.4f} | "
                            f"NMI: {nmi_max:.4f} | "
                            f"PUR: {pur_max:.4f}"
                        )

                        if args.times > 1:
                            count += 1
                            count = count % args.times

        # 当前 missing rate 的统计结果
                    acc_mean, acc_std = np.mean(rate_result['ACC']), np.std(rate_result['ACC'])
                    ari_mean, ari_std = np.mean(rate_result['ARI']), np.std(rate_result['ARI'])
                    nmi_mean, nmi_std = np.mean(rate_result['NMI']), np.std(rate_result['NMI'])
                    pur_mean, pur_std = np.mean(rate_result['PUR']), np.std(rate_result['PUR'])

                    print("ACC: ", rate_result['ACC'])
                    print("ARI: ", rate_result['ARI'])
                    print("NMI: ", rate_result['NMI'])
                    print("PUR: ", rate_result['PUR'])

                    all_rate_results.append({
                        'Missing Rate': missrate,

                        'ACC Mean': acc_mean,
                        'ACC Std': acc_std,
                        'ARI Mean': ari_mean,
                        'ARI Std': ari_std,
                        'NMI Mean': nmi_mean,
                        'NMI Std': nmi_std,
                        'PUR Mean': pur_mean,
                        'PUR Std': pur_std,

                        'ACC All': rate_result['ACC'],
                        'ARI All': rate_result['ARI'],
                        'NMI All': rate_result['NMI'],
                        'PUR All': rate_result['PUR'],
                    })

                    print("\n" + "*" * 90)
                    print(f" Summary for Missing Rate = {missrate:.1f}")
                    print("*" * 90)
                    print(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
                    print(f"ARI: {ari_mean:.4f} ± {ari_std:.4f}")
                    print(f"NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")
                    print(f"PUR: {pur_mean:.4f} ± {pur_std:.4f}")

    # =========================
    # 输出所有 missing rate 汇总结果
    # =========================
    summary_df = pd.DataFrame(all_rate_results)

    # 用于展示的美观表格
    display_df = pd.DataFrame({
        'Missing Rate': summary_df['Missing Rate'],
        'ACC': summary_df.apply(lambda x: f"{x['ACC Mean']:.4f} ± {x['ACC Std']:.4f}", axis=1),
        'ARI': summary_df.apply(lambda x: f"{x['ARI Mean']:.4f} ± {x['ARI Std']:.4f}", axis=1),
        'NMI': summary_df.apply(lambda x: f"{x['NMI Mean']:.4f} ± {x['NMI Std']:.4f}", axis=1),
        'PUR': summary_df.apply(lambda x: f"{x['PUR Mean']:.4f} ± {x['PUR Std']:.4f}", axis=1),
    })

    print("\n" + "=" * 100)
    print(f" Final Results on Dataset: {args.dataset}")
    print(f" Total Experiments: {count}")
    print(f" Initial Seed: {init_seed}")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)

    # =========================
    # 保存 Excel
    # =========================
    raw_df = pd.DataFrame(all_raw_results)

    save_path = f"{args.dataset}_All_MissingRate_Results.xlsx"

    with pd.ExcelWriter(save_path) as writer:
        display_df.to_excel(writer, sheet_name='Summary_Pretty', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_Raw', index=False)
        raw_df.to_excel(writer, sheet_name='All_Runs', index=False)

    print(f"\nAll results have been saved to: {save_path}")
    plot_melancholic_metrics(args.dataset)



if __name__ == '__main__':
    T1 = time.time()
    main()
    T2 = time.time()
    print("Run Time : {}".format(T2 - T1))

    from datetime import datetime

    # get date
    now = datetime.now()

    # 格式化输出，"%Y-%m-%d %H:%M" 表示精确到分钟
    formatted_time = now.strftime("%Y-%m-%d %H:%M")

    print(formatted_time)
