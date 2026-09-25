import time
import collections

import numpy as np
import logging

logging.getLogger('matplotlib.font_manager').disabled = True
from units.config import *
from train import *


def main():
    args = parse_args()

    T = args.times
    config = dataset_config(args)


    if args.times > 1:
        seeds = [5, 15, 25, 45, 50]

    else:
        seeds = [45]

    missrate = args.missing_rate

    all_rate_results = []

    all_raw_results = collections.defaultdict(list)
    count = 0
    init_seed = seeds

    print(f'lamda_1: {args.lamda_1}, lamda_2: {args.lamda_2}, lamda_3: {args.lamda_3}')
    print("\n" + "=" * 90)
    print(f" Missing Rate = {missrate:.1f}")
    print("=" * 90)

    args.missing_rate = missrate

    rate_result = collections.defaultdict(list)

    for i in range(T):
        print("\n" + "-" * 70)
        print(f"Running Experiment {count + 1}")
        print(f"Missing Rate : {missrate:.1f}")
        print(f"Repeat Index : {i + 1}/{T}")

        args.seed = seeds[count]
        print(f"Seed         : {args.seed}")

        Result, Loss = Training(args, config)

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
        all_raw_results['ACC'].append(acc_max)
        all_raw_results['ARI'].append(ari_max)
        all_raw_results['NMI'].append(nmi_max)
        all_raw_results['PUR'].append(pur_max)


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



    summary_df = pd.DataFrame(all_rate_results)

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

    # raw_df = pd.DataFrame(all_raw_results)
    #
    # save_path = f"{args.dataset}_All_MissingRate_Results.xlsx"
    #
    # with pd.ExcelWriter(save_path) as writer:
    #     display_df.to_excel(writer, sheet_name='Summary_Pretty', index=False)
    #     summary_df.to_excel(writer, sheet_name='Summary_Raw', index=False)
    #     raw_df.to_excel(writer, sheet_name='All_Runs', index=False)
    #
    # print(f"\nAll results have been saved to: {save_path}")
    # plot_melancholic_metrics(args.dataset)



if __name__ == '__main__':
    T1 = time.time()
    main()
    T2 = time.time()
    print("Run Time : {}".format(T2 - T1))

    from datetime import datetime

    now = datetime.now()

    formatted_time = now.strftime("%Y-%m-%d %H:%M")

    print(formatted_time)
