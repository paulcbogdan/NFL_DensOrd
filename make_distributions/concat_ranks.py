import os

import numpy as np
import scipy
from tqdm import tqdm

from make_distributions.smooth import smooth_horizontal
from make_distributions.utils import PKL_CACHE, pickle_wrap


def concat_all_rank_neighbors(df, ar, max_rank, ar_year, num_neighbors_concat,
                              df_expert, expert, pos, x_tiles,
                              concat_correction=1.0):
    '''
    Applies the concatenation described in the white paper. It works by looping
        through all of the ranks, and then doing the concatenation for each rank
        using concat_rank_w_neighbors(...)
    
    :param df: input dataframe
        Corresponds to df_train in setup_expert_distributions.py
    :param ar: matrix derived from the input dataframe, see density.py
    :param max_rank: Highest (worst) rank analyzed. A limit
    :param ar_year: Matrix of identical shape to ar, which specifies the year of
        each entry in ar.
    :param num_neighbors_concat: Number of neighbors to concatenate with
    :param df_expert: Variable used in density.py. Modification of df
    :param expert: Name of expert used. Often 'mean_expert'
    :param pos: Position (str)
    :param x_tiles: Cumulative density is calculated discretely for the
         0th percentile, 0.1th percentile, etc. x_tiles specifies the
         discrete percentiles used.
    :param concat_correction: If non-zero, a concatenation correction is applied
        See the white paper for details on why biases arise at high ranks.
        Although this specific correction is not described in the white paper
        it is similar to the correction used for vertical smoothing, which
        was described. Concat_correction should be a float from [0.0, 1.0],
        where 0.0 = no correction and 1.0 = full correction.
    :return: 
    '''
    if num_neighbors_concat:
        _, effect_if_worse_abs = get_pt_rank_booster_pkl(df_expert, expert, pos,
                                                         x_tiles, df)
    else: effect_if_worse_abs = None

    if max_rank is None: max_rank = ar.shape[0]
    ar_added = []
    ar_year_added = []
    for rank0 in tqdm(range(0, max_rank), desc='get_rank_pts_to_density'):
        ys_span = list(ar[rank0, :])
        l_year = list(ar_year[rank0, :])
        # combines rank i with nearby ranks
        #   (e.g., with i-5 to i-1 and with i+1 to i+5)
        #   applies correction via effect_if_worse_abs
        if num_neighbors_concat:
            ys_span, l_year = concat_rank_w_neighbors(rank0, ar, ys_span,
                                                      effect_if_worse_abs,
                                                      num_neighbors_concat,
                                                      l_year, ar_year,
                                apply_correction=abs(concat_correction) > .0001,
                                concat_correction=concat_correction)
        ar_added.append(ys_span)
        ar_year_added.append(l_year)
    ar = np.array(ar_added)
    ar_year = np.array(ar_year_added)
    return ar, ar_year


def concat_rank_w_neighbors(rank0, ar, ar_rank0, effect_if_worse_abs,
                            num_neighbors_concat, l_year, ar_year,
                            apply_correction=True,
                            concat_correction=1.0):
    '''
    Concatenates the rank0 data with data from nearby ranks. 
    
    :param rank0: Center rank to concatenate around
    :param ar: Matrix (array) of data, see density.py
    :param ar_rank0: rank0 vector from ar 
    :param effect_if_worse_abs: Object used to apply concatenation correction
    :param num_neighbors_concat: Number of neighboring ranks to concatenate
        rank0 is concatenated with:
        [rank0-num_neighbors_concat, rank0+num_neighbors_concat]
    :param l_year: Year corresponding to time t for rank0
    :param ar_year: Year corresponding to time t for ar
    :param apply_correction: If True, a concatenation correction is applied.
        See the white paper for details on why biases arise at high ranks 
    :param concat_correction: Amount of correction, float from [0.0, 1.0]
    :return: Updated data vector (ar_rank0) and year matrix (l_year)
    '''
    for dif in range(-num_neighbors_concat, num_neighbors_concat + 1):
        if dif == 0: continue
        rank0_i = rank0 + dif
        if rank0_i >= 0 and rank0_i < ar.shape[0]:
            ar_i = ar[rank0_i, :]
            ar_year_i = ar_year[rank0_i, :]
            n_nan = np.count_nonzero(np.isnan(ar_i))
            ar_year_i = ar_year_i[~np.isnan(ar_i)]
            ar_i = ar_i[~np.isnan(ar_i)]
            if apply_correction and rank0 < num_neighbors_concat and \
                    rank0_i > 2*rank0:
                ar_rank0 += apply_concat_correction(ar_i, rank0, rank0_i,
                                                    effect_if_worse_abs,
                                                    weight=concat_correction)
            else:
                ar_rank0 += list(ar_i)
            l_year += list(ar_year_i)

            ar_rank0 += [np.nan] * n_nan
            l_year += [np.nan] * n_nan
        else:
            ar_rank0.extend([np.nan] * ar.shape[1])
            l_year.extend([np.nan] * ar.shape[1])
    return ar_rank0, l_year

def get_pt_rank_booster_pkl(df_expert, expert, pos, x_tiles, df):
    '''
    Wrapper function for get_pt_rank_booster(...), which uses a pickle cache
    
    Gets a variable used for the concatenation correction. 
        Not described in the white paper but this procedure is similar to 
        vertical smoothing correction.
    '''
    fp_pt_rank_booster = get_pt_rank_fp(df_expert, x_tiles, expert, pos)
    effect_if_worse_dif, effect_if_worse_abs = pickle_wrap(
        fp_pt_rank_booster,
        lambda: get_pt_rank_booster(df, expert, pos, x_tiles))
    return effect_if_worse_dif, effect_if_worse_abs

def get_pt_rank_fp(df_pivot, x_tiles, expert, pos):
    '''
    Gets the file path for the pickle cache of get_pt_rank_booster(...)
    '''
    year_st = np.min(df_pivot.columns.get_level_values(0))
    year_end = np.max(df_pivot.columns.get_level_values(0))
    str_x_tile = f'{x_tiles[0]}-{x_tiles[-1]}-{len(x_tiles)}'
    fn = f'pt_rank_booster_{expert}_{pos}_{year_st}_{year_end}_{str_x_tile}.pkl'
    return os.path.join(PKL_CACHE, fn)

def get_pt_rank_booster(df, expert, pos, x_tiles, do_quick=False):
    '''
    Gets a variable used for the concatenation correction. 
        Not described in the white paper but this procedure is similar to 
        vertical smoothing correction.
    '''

    ar_rank_tile = get_cum_dist_T(df, expert, pos, x_tiles)
    if not do_quick: ar_rank_tile = smooth_horizontal(ar_rank_tile, 50,
                                                      'nearest')

    # rank i-1 [better] minus rank i. i.e., improvement of being higher rank
    ar_d1 = get_percentile_to_pts_gradient(ar_rank_tile)
    if not do_quick: ar_d1 = scipy.ndimage.gaussian_filter(ar_d1,
                                    (6, ar_d1.shape[1]/300)) # cross validate!!

    effect_if_worse_dif = {}
    effect_if_worse_abs = {}

    for input_rank in range(ar_d1.shape[0]):
        for tile in range(ar_d1.shape[1]):
            baseline_pt = ar_rank_tile[input_rank, tile]
            rolling_sum = 0
            if (input_rank, int(baseline_pt), 1) in effect_if_worse_dif:
                continue
            for rank_dif in range(1, 5):
                worse_rank = input_rank + rank_dif
                if worse_rank >= ar_d1.shape[0]: continue
                rolling_sum += ar_d1[worse_rank, tile]
                effect_if_worse_dif[(input_rank, int(baseline_pt),
                                     rank_dif)] = -rolling_sum
                effect_if_worse_abs[(input_rank, int(baseline_pt),
                                     worse_rank)] = -rolling_sum

        # sometimes these won't be filled in by the actual input_data
        for pt in list(range(5)):
            for rank_dif in range(5):
                if (input_rank, int(pt), rank_dif) not in effect_if_worse_dif:
                    effect_if_worse_dif[(input_rank, int(pt),
                                         rank_dif)] = 0
                    effect_if_worse_abs[(input_rank, int(pt),
                                         input_rank+rank_dif)] = 0

        # sometimes these won't be filled in by the actual input_data
        for pt in list(range(40, 56)):
             for rank_dif in range(5):
                 if (input_rank, int(pt), rank_dif) not in effect_if_worse_dif \
                        and (input_rank, pt-1, rank_dif) in effect_if_worse_dif:
                     effect_if_worse_dif[(input_rank, int(pt), rank_dif)] = \
                         effect_if_worse_dif[(input_rank, pt-1, rank_dif)]
                     effect_if_worse_abs[(input_rank, int(pt),
                                          input_rank+rank_dif)] = \
                         effect_if_worse_dif[(input_rank, pt-1, rank_dif)]

    effect_if_worse_abs = interpolate_effect_if(effect_if_worse_abs)
    return effect_if_worse_dif, effect_if_worse_abs


def get_percentile_to_pts_gradient(ar):
    '''
    Gets a variable used for the concatenation correction. 
        Not described in the white paper but this procedure is similar to 
        vertical smoothing correction.
    '''
    padding = np.empty((1, ar.shape[1])) # for players who were not ranked 
                                         # (nan turned to 0 above)
    padding[:] = np.nan
    pad_bottom = np.append(ar, padding, axis=0)[1:, :]
    ar = ar - pad_bottom
    ar = ar[:-1, :]
    return ar


def apply_concat_correction(l, rank0_target, rank0_i, effect_if_worse_abs,
                            weight=1.0):
    '''
    Applies the concatenation correction. Not in white paper.
    '''
    # if the target is better than the rank_i, then make rank_i better
    if rank0_i > rank0_target:
        return map(lambda x: x -
            weight*effect_if_worse_abs[(rank0_target, int(x), rank0_i)], l)
    else: #if the target is wrose than the rank_i, then make rank_i worse
        return map(lambda x: x +
            weight*effect_if_worse_abs[(rank0_i, int(x), rank0_target)], l)


def interpolate_effect_if(effect_d):
    '''
    Used for the neighbor concatenation correction. Not in white paper.
    '''
    print('Interpolating the pt booster')
    from scipy.interpolate import griddata as gd
    keys = np.array(list(effect_d.keys()))
    vals = np.array(list(effect_d.values()))
    Xi, Yi, Zi = np.meshgrid(range(0, max(keys[:, 0]) + 10),
                             range(-10, max(keys[:, 1]) + 30),
                             range(0, max(keys[:, 2]) + 10))
    Xi = Xi.flatten()
    Yi = Yi.flatten()
    Zi = Zi.flatten()

    x, y, z = keys.transpose()
    vals_out = gd((x, y, z), vals, (Xi, Yi, Zi), method='nearest')
    for xi, yi, zi, val in zip(Xi, Yi, Zi, vals_out):
        effect_d[(xi, yi, zi)] = val
    return effect_d


def interpolate_effect_if_d(effect_d, num_dims=4):
    '''
    Used for the neighbor concatenation correction. Not in white paper.
    '''
    print(f'Interpolating the booster: num_dims = {num_dims}')
    from scipy.interpolate import griddata as gd
    keys = np.array(list(effect_d.keys()))
    vals = np.array(list(effect_d.values()))
    ranges = [range(min(keys[:, i])-2,
                    max(keys[:, i]) + 30) for i in range(num_dims)]
    keys_grid = np.meshgrid(*ranges)
    for i in range(len(keys_grid)):
        keys_grid[i] = keys_grid[i].flatten()

    keys_grid = np.array(keys_grid).transpose()
    vals_out = gd(keys, vals, keys_grid, method='nearest')
    for key, val in zip(keys_grid, vals_out):
        effect_d[tuple(key)] = val

    return effect_d


def get_cum_dist_T(df, expert, pos, x_tiles, max_rank=None):
    '''
    Used for the neighbor concatenation correction. Not in white paper.
    '''

    df_ = df.loc[pos, :, :, :]
    df_.dropna(subset=[expert], inplace=True)
    df_pivot = df_.pivot(index=expert, columns=['year_', 'week_'],
                         values='points')
    ar = np.array(df_pivot)
    ar.sort(axis=1)
    ar_out = []
    if max_rank is None: max_rank = ar.shape[0]
    for rank0 in range(0, max_rank):
        y = ar[rank0, :]
        y = y[~np.isnan(y)]
        if len(y) < 15: continue
        n_weeks = len(y)
        x_orig = np.linspace(1/n_weeks, 1-1/n_weeks, n_weeks)
        spl = scipy.interpolate.interp1d(x_orig, y, kind='nearest',
                                         bounds_error=False,
                                         fill_value='extrapolate')
        y = spl(x_tiles)
        ar_out.append(y)
    return np.array(ar_out)
