from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as stats
from tqdm import tqdm

'''
This file just contains some functions used to process the expert rankings
Not commented
'''


def get_experts_w_enough_ranks(df, experts, min=None):
    expert_to_cnt = {}
    for expert in experts:
        expert_to_cnt[expert] = \
            (df[expert].groupby(level=(1, 2)).sum() > 0).sum()
    if min is None: min = max(expert_to_cnt.values())
    elif min < 1: min = max(expert_to_cnt.values()) * min
    return [expert for expert, cnt in expert_to_cnt.items() if cnt >= min]


def get_best_experts(df, experts, tile=.40, do_plot=False):
    if 'true_rank' not in df.columns: add_true_ranking(df)

    expert_rs_d = defaultdict(list)
    print(f'Number of experts: {len(experts)=}')
    for year_week, df_week in tqdm(df[df['true_rank'] <= 47].groupby(
            level=(1, 2)), desc='Calculating expert accuracies (rho)'):
        for expert in experts:
            df_week_expert = df_week[['true_rank', expert]].fillna(47)
            r, p = stats.spearmanr(df_week_expert['true_rank'],
                                   df_week_expert[expert])
            expert_rs_d[expert].append(r)
    expert_rs_d = dict([(expert, np.nanmean(rs)) for expert,
                                                     rs in expert_rs_d.items()])
    expert_rs_l = sorted(list(expert_rs_d.values()))
    if do_plot:
        plt.hist(expert_rs_l)
        plt.show()
    r_cutoff = expert_rs_l[int(len(expert_rs_d) * tile)]
    return [expert for expert, r in expert_rs_d.items() if r > r_cutoff - .0001]


def add_true_ranking(df):
    df['true_rank'] = df['points'].groupby(
        level=(0, 1, 2)).rank(method='first', ascending=False)


def add_mean_expert(df_rankings, experts, drop_experts=True):
    print(f'Number of experts averaging: {len(experts)=}')
    df_rankings['mean_expert_float'] = df_rankings[experts].mean(axis=1)
    df_rankings['mean_expert_std'] = df_rankings[experts].std(axis=1)
    df_rankings['mean_expert'] = df_rankings['mean_expert_float'].groupby(
        level=(0, 1, 2)).rank(method='first')
    if drop_experts:
        df_rankings.drop(columns=experts, inplace=True)