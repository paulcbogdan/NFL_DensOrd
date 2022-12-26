import numpy as np


def get_test_accuracy(df_test, ED, min_rank=0, max_rank=48, eval_float=False,
                      pos='RB'):
    '''
    Evaluate how accuracy is the created Expert Distribution (ED). Simply done
        by probability density associated with a player's actual points (Y),
        given their rank. Evaluated based on a testing set. Accuracy is assessed
        as log-likelihood.
    :df_test: testing date
    :ED: Expert Distribution
    :min_rank: Minimum (Best) rank to consider
    :max_rank: Maximum (Worst) rank to consider
        :param eval_float: If True, then testing accuracy is judged slightly
        differently. Players may be treated as being between two ranks
        (e.g., a player may be ranked 5.5). If True, it just calls
        get_test_accuracy_float_ranks(...), which is similar to the present
        function
    :pos: Position (str)
    :return: (float) Accuracy score & size of the test set (number of examples)

    '''
    print(f'Experts: {ED.experts}')
    df_test = df_test[(df_test.index.get_level_values(0) == ED.pos) &
                      (df_test[ED.experts].min(axis=1) > min_rank) &
                      (df_test[ED.experts].max(axis=1) < max_rank)]
    if eval_float: return get_test_accuracy_float_ranks(df_test, ED, max_rank)

    df_test.fillna(0, inplace=True)
    X = np.array(df_test[ED.experts]).astype(int)
    Y = df_test['points']
    if pos == 'DST':
        Y_idx = ((Y + 10)/ 50 * 2500).astype(int)
    else:
        Y_idx = (Y / 50 * 2500).astype(int)
    Y_idx[Y_idx >= ED.ar_all_density.shape[2]] = ED.ar_all_density.shape[2]-1
    padding = np.full((ED.ar_all_density.shape[0], 1, ED.ar_all_density.shape[2]),
                      np.log(1/ED.ar_all_density.shape[2]))
    ar_pad = np.concatenate([padding, ED.ar_all_density], axis=1)

    # Baseline would be the accuracy of a uniform distribution
    baseline_score_l = np.log(1/ED.ar_all_density.shape[2])
    total_score = 0
    test_size = 0
    scores = []
    for x, y_idx, y in zip(X, Y_idx, Y):
        player_vals = np.diag(ar_pad[:, x, y_idx])
        ll = player_vals.mean()
        bf = ll - baseline_score_l
        total_score += bf
        scores.append(total_score)
        test_size += 1
    print(f'{ED.pos} | Overall score: {total_score:.3f} (n = {test_size})')
    print(f'Experts: {ED.experts}')
    return total_score, test_size


def get_test_accuracy_float_ranks(df_test, ED, max_rank=48):
    '''
    Similar to get_test_accuracy(...), but here testing accuracy is judged
        slightly differently. Players may be treated as being between two ranks
        (e.g., a player may be ranked 5.5).
    :df_test: testing date
    :ED: Expert Distribution
    :return: (float) Accuracy score & size of the test set (number of examples)
    '''

    df_test.fillna(0, inplace=True)
    X = np.array(df_test[['mean_expert_float', 'mean_expert']])
    Y = df_test['points']
    Y_idx = (Y / 50 * 1000).astype(int)
    Y_idx[Y_idx >= ED.ar_all_density.shape[2]] = ED.ar_all_density.shape[2] - 1
    padding = np.full((ED.ar_all_density.shape[0], 1, ED.ar_all_density.shape[2]),
                      np.log(1 / ED.ar_all_density.shape[2]))
    ar_pad = np.concatenate([padding, ED.ar_all_density], axis=1)
    baseline_score_l = np.log(1 / ED.ar_all_density.shape[2])
    total_score = 0
    cnt = 0
    for x, y_idx, y in zip(X, Y_idx, Y):
        x[x >= max_rank] = max_rank
        x[1] = (x[0] + x[1])/2
        x_high = np.ceil(x).astype(int)
        x_low = np.floor(x).astype(int)
        weight_high = 1-(x_high-x)
        weight_low = 1-(x-x_low)
        player_vals = weight_low*np.exp(np.diag(ar_pad[:, x_low, y_idx])) + \
                      weight_high*np.exp(np.diag(ar_pad[:, x_high, y_idx]))
        player_vals = np.log(player_vals)
        ll = player_vals.mean()
        bf = ll - baseline_score_l
        total_score += bf
        cnt += 1
        print(f'{x=} | {ll=:.3f} | {bf=:.3f} | {total_score=:.3f}')
    print(f'{ED.pos} | Overall score: {total_score:.3f} (n = {cnt})')
    print(f'Experts: {ED.experts}')
    return total_score, cnt
