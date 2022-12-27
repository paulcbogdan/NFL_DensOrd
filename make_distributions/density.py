from multiprocessing import Pool

import numpy as np
from colorama import Fore
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from make_distributions import plot as plot
from make_distributions.concat_ranks import concat_all_rank_neighbors
from make_distributions.smooth import smooth_horizontal, \
    smooth_vertical


def get_density_array(df, experts_used, pos, x_density_pts, x_tiles,
                      max_rank=None,
                      num_neighbors_concat=1, do_CV_density=False,
                      horz_smooth_amt=False, concat_correction=1.0,
                      vert_smooth_amt=2,
                      vert_smooth_correct=False):
    '''
    Calculates the density array.
        See setup_expert_distributions.get_expert_distribution
    :param df: Dataframe containing all the ranking and scores data
    :param experts_used: Expert names whose rankings are used. Usually just
        ['mean_expert']
    :param pos: Position (str)
    :param x_density_pts:
    :param x_tiles: Cumulative density is calculated discretely for the
         0th percentile, 0.1th percentile, etc. x_tiles specifies the
         discrete percentiles used.
    :param max_rank: Highest (worst) rank analyzed. A limit
    :param num_neighbors_concat: Number of neighbors to concatenate with
    :param do_CV_density: If True, cross-validation will be used to find the
        optimal bandwidth density for kernel density estimation? If not, it will
        use a constant, which is a default parameter in another function
    :param horz_smooth_amt:
    :param concat_correction:
    :param vert_smooth_amt:
    :param vert_smooth_correct:
    :return: ar_density, a 2D matrix representing all the density distributions
        Corresponds to matrix P' on page 4 of the white paper
    '''
    print(f'{Fore.LIGHTGREEN_EX}get_rank_pts_to_density{Fore.RESET}'
          f'(df, expert={Fore.LIGHTRED_EX}{experts_used}{Fore.RESET},'
          f' pos={pos}, ..., max_rank={max_rank}, '
          f'concat_proximal_ranks={num_neighbors_concat}, '
          f'CV_density={do_CV_density}')

    # Takes the dataframe input and coverts it into a matrix (ar) where:
    #  - each row is an expert rank
    #  - each column is a timepoint (week, year)
    #  - ar[i, t] correspond to the ith ranked player's points scored at time t
    df_expert = df.loc[pos, :, :, :].dropna(subset=[experts_used])
    df_expert = df_expert[df_expert[experts_used] < max_rank + 5]
    df_pivot = df_expert.pivot(index=experts_used, columns=['year_', 'week_'],
                               values='points')
    ar, ar_year = get_array_from_df(df_pivot)
    ar, ar_year = concat_all_rank_neighbors(df, ar, max_rank, ar_year,
                                            num_neighbors_concat, df_expert,
                                            experts_used, pos, x_tiles,
                                            concat_correction=concat_correction)

    # Based on the data matrix (ar), fit the kernel density estimates
    if do_CV_density:
        ar_density = fit_density_CV_bandwidth(ar, x_density_pts, ar_year,
                                              max_rank=max_rank)
    else:
        ar_density = fit_density_constant_bandwidth(ar, x_density_pts,
                                                    max_rank=max_rank,
                                                    bandwidth=1)
    ar_density = np.exp(ar_density)

    # Smooth and normalize the density estimates
    if horz_smooth_amt: ar_density = smooth_horizontal(ar_density,
                                                       size=horz_smooth_amt,
                                                       mode='nearest')
    ar_density = normalize_horizontal(ar_density)
    if vert_smooth_amt: ar_density = \
        smooth_vertical(ar_density, size=vert_smooth_amt, mode='nearest',
                        correct=vert_smooth_correct)
    ar_density = normalize_horizontal(ar_density)

    return ar_density


def get_array_from_df(df_pivot, rand_mult=5):
    '''
    Takes the dataframe input and coverts it into a matrix (ar) where:
       - each row is an expert rank
       - each column is a timepoint (week, year)
       - ar[i, t] correspond to the ith ranked player's points scored at time t
    Also creates ar_year, which is a matrix of identical shape to ar, which
        specifies the year of each entry in ar. This is used for a
        cross-validation step, where the data are split into folds based on
        their year.
        ar_year[i, t] corresponds to the year of ar[i, t]

    :param df: Input dataframe
        Corresponds to df_train in setup_expert_distributions.py
    :param pos:
    :param expert:
    :param max_rank: Highest (worst) rank analyzed. A limit
    :return: ar, ar_year
    '''
    ar = np.array(df_pivot)
    ar_year = np.repeat(df_pivot.columns.get_level_values(0)[:, np.newaxis],
                        ar.shape[0], axis=1).transpose()
    if rand_mult > 1:
        ar, ar_year = jiggle_array(ar, ar_year, rand_mult)
    return ar, ar_year


def jiggle_array(ar, ar_year, rand_mult):
    '''
    Since the input data are discrete (e.g., with 11.1 pts, and 11.2 pts, but
        nothing in between), this will create issues for the kernel density
        estimation. To fix this, we will add a small amount of random noise to
        the data, so that the data are no longer discrete. The data are also
        duplicated, each with a different random noise added.
    If rand_mult=5, then one element of ar containing 11.1 will be changed
        into five elements in ar_jiggled. If ar is an I x T matrix, and
        ar_jiggled is an I x (5*T) matrix.

    :param ar: Data matrix, see get_array_from_df(...)
    :param x_pts: Discrete points represented by columns of ar
    :param ar_year: Year matrix, see get_array_from_df(...). ar_year[i, t]
        corresponds to the year for ar[i, t]
    :return: ar_jiggled, ar_year_repeated
    '''

    ar_repeated = np.repeat(ar, rand_mult, axis=1)
    ar_jiggled = ar_repeated + np.random.uniform(-0.05, 0.05,
                                                 size=ar_repeated.shape)
    ar_year_repeated = np.repeat(ar_year, rand_mult, axis=1)
    return ar_jiggled, ar_year_repeated


def fit_density_constant_bandwidth(ar, x_pts, max_rank=None, bandwidth=0.8,
                                   exp=False, kernel='exponential'):
    '''
    Calculates kernel density while using a constant kernel bandwidth for
        every rank.
    The code below refers to 'rank0' rather than 'rank' to emphasize that the
        rank is 0-indexed, which differs from how fantasy ranks are typically
        reported (1-indexed) and how they are described in the white paper.

    :param ar: Data matrix, see get_array_from_df(...)
    :param x_pts: Discrete points represented by columns of ar
    :param max_rank: Highest (worst) rank analyzed
    :param bandwidth: Constant width
    :param exp: If true, then the density is exponentiated
    :param kernel: Kenrel type
    :return: ar_density (kernel density represented as a matrix)
    '''
    if max_rank is None: max_rank = ar.shape[0]
    ar_density = []
    for rank0 in range(0, max_rank):
        y = ar[rank0, :]
        y = y[~np.isnan(y)]
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

        kde.fit(np.expand_dims(y, axis=1))
        densities = kde.score_samples(np.expand_dims(x_pts, axis=1))
        if exp: densities = np.exp(densities)
        ar_density.append(densities)
    ar_density = np.array(ar_density)
    if exp: ar_density = normalize_horizontal(ar)
    return ar_density


def fit_density_CV_bandwidth(ar, x_pts, ar_year, max_rank=None,
                             exp=False, kernel='exponential',
                             do_plot=False):
    '''
    Calculates kernel density while using a constant kernel bandwidth for
        every rank.

    :param ar: Data matrix, see get_array_from_df(...)
    :param x_pts: Discrete points represented by columns of ar
    :param ar_year: Year matrix, see get_array_from_df(...). ar_year[i, t]
        corresponds to the year for ar[i, t]
    :param max_rank: Highest (worst) rank analyzed
    :param bandwidth: Constant width
    :param exp: If true, then the density is exponentiated
    :param kernel: Kenrel type
    :return: ar_density (kernel density represented as a matrix)
    '''
    if max_rank is None: max_rank = ar.shape[0]
    ar_density = []
    iterable = [(ar_year[rank0, :], ar[rank0, :], rank0, kernel)
                for rank0 in range(0, max_rank)]

    with Pool(6) as P:
        print('Multiprocessing map')
        grids = P.starmap(get_optimal_rank_bandwidth, tqdm(iterable,
                                               desc='multiprocess CV density'))
    for grid, rank0 in grids:
        densities = grid.score_samples(np.expand_dims(x_pts, axis=1))
        if exp:
            densities = np.exp(densities)
        print(f'Best params confirm: {rank0}: {grid.best_params_}')
        ar_density.append(densities)
    ar_density = np.array(ar_density)
    if exp: ar_density = normalize_horizontal(ar)
    if do_plot: plot.plot_density(np.log(ar_density), x_pts)
    return ar_density


def get_optimal_rank_bandwidth(groups, y, rank0, kernel='exponential'):
    '''
    Finds optimal bandwidth for density estimation of a specific rank, using
        cross-validation
    Note that this function uses GroupKFold cross-validation, which organizes
        the data into folds based on the year of the data. These year data
        are managed by ar_year in the fit_density_CV_bandwidth(...) which calls
        the present function.

    :param groups: Year of each entry in y (ar_year[rank0, :])
    :param y: Data for a specific rank (ar[rank0, :])
    :param rank0: Rank of y
    :param kernel: Kernel type
    :return: grid (GridSearchCV object) and the unchanged rank0 again
    '''
    groups = groups[~np.isnan(y)]
    y = y[~np.isnan(y)]
    param_space = np.exp(np.linspace(-3, 1, 15))
    kde = KernelDensity(kernel=kernel)
    params = {'bandwidth': param_space}
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_groups)
    grid = GridSearchCV(kde, params, cv=gkf, verbose=0)
    grid.fit(np.expand_dims(y, axis=1), groups=groups)
    print(f'Best params: {rank0}: {grid.best_params_}')
    return grid, rank0


def normalize_horizontal(ar_density):
    '''
    Normalizes the horizontal axis of an array so that each row sums to 1.

    :param ar_density: a 2D matrix representing all the density distributions
        Corresponds to matrix P' on page 4 of the white paper
    '''
    assert np.min(ar_density) > -.00000001, 'Can\'t normalize log-scaled ar'
    return ar_density / ar_density.sum(axis=1)[:, None]
