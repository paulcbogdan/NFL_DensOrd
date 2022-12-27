import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from colorama import Fore
from tqdm import tqdm

from scrape_prepare_input.organize.organize_input import get_df_rankings_scores
from make_distributions.test_accuracy import get_test_accuracy
from make_distributions.pd_utils import get_train_test, get_train_test_single_year
from make_distributions.expert_processing import get_best_experts, add_mean_expert
from make_distributions.density import get_density_array
from make_distributions.expert_distribution import Expert_Distribution
from make_distributions.plot import save_ED_plot
from make_distributions.utils import pickle_wrap, PKL_CACHE


np.random.seed(0)


def load_data(year_st=2013, year_end=2021):
    '''
    Loads data scraped and organized in get_df_rankings_scores(...).
        The dataframe return contains the rankings and performances (scores)
        for each player of every week from year_st to year_end.
    The multiindex dataframe is described in get_df_rankings_scores(...)

    Note that pickle_wrap(...) is used. This is a convenient wrapper. The first
        time get_df_rankings_scores(...) is called, the wrapper will save the
        results to a passed filepath. Later times it is called, the wrapped
        will automatically load the saved results, unless easy_override=True

    :param year_st: First year of data to load
    :param year_end: Last year of data to load, inclusive
    :return: Multiindex dataframe
    '''
    fp_pkl_overall = os.path.join(PKL_CACHE,
                                  f'df_rankings_scores_{year_st}_{year_end}.pkl')
    df_rankings, df_scores = pickle_wrap(fp_pkl_overall,
                                         lambda: get_df_rankings_scores(
                                             year_st, year_end),
                                             easy_override=False)
    return df_rankings


def fit_and_predict_testing(df_rankings, pos, train_st=2013, train_end=2020,
                            test_st=2020, test_end=2021, single_year_test=False,
                            best_experts_cutoff=0.2, concat_correction=1.0,
                            min_rank=0, max_rank=8, do_mean=True,
                            do_CV_density=False, horz_smooth_amt=False,
                            concat_proximal_ranks=4, eval_float=True,
                            density_resolution=2500, vert_smooth_amt=None,
                            vert_smooth_correct=False, plot_name='',
                            do_test=True):
    '''
    Fits the kernel density distributions, which are handled by an
        Expert_Distribution object. This function then evaluates the testing
        accuracy of the density distribution. Finally, the function plots
        the distribution.

    Distributions fit are specific to a position (pos).

    Some comments are provided below, dividing up the function into major steps

    :param df_rankings: Input dataframe. See load_data(...)
    :param pos: Position (str)
    :param train_st: First year of training data
    :param train_end: Last year of training data, inclusive
    :param test_st: First year of testing data
    :param test_end: Last year of testing data, inclusive
    :param single_year_test: If you only want to test on one year, this can be
        used and then only test_st is used for testing, and all other years
        train_st and later are used for training
    :param best_experts_cutoff: Float between 0 and 1. This is the cutoff for
        the best experts to use. For example, if 0.2, then the top 20% of
        experts are used.
    :param concat_correction: If non-zero, a concatenation correction is applied.
        See the white paper for details on why biases arise at high ranks.
        Although this specific correction is not described in the white paper
        it is similar to the correction used for vertical smoothing, which
        was described. Concat_correction should be a float from [0.0, 1.0],
        where 0.0 = no correction and 1.0 = full correction.
    :param min_rank: Lowest (best) rank analyzed
    :param max_rank: Highest (worst) rank analyzed
    :param do_mean: If True, then the mean of the experts is used
    :param do_CV_density: If True, uses cross-validation to find the optimal
        kernel density bandwidth
    :param horz_smooth_amt: Degree of smoothing applied horizontally.
        Larger = more smoothing. No upper limit.
    :param concat_proximal_ranks:
    :param eval_float: If True, then testing accuracy is judged slightly
        differently. Players may be treated as being between two ranks
        (e.g., a player may be ranked 5.5).
    :param density_resolution: Density of the distribution. Higher = more
        dense. This is the number of discrete points in the distribution.
    :param vert_smooth_amt: Degree of smoothing applied vertically.
        Larger = more smoothing. No upper limit.
    :param vert_smooth_correct: If True, applies a correction for the vertical
        smoothing to prevent edge biases. Similar to concat correction
    :param plot_name: String to be added to the plot filename
    :return: test_acc (accuracy, log likelihood)
    :return: test_size (size of test set; number of examples)
    '''
    def organize_params(d):
        '''
        Organizes the parameters into a string that can either be used
            1. to print out all the parameters nicely with colors
                (helps if you are interested in the print output)
            2. to be used in the plot filename
        :param d: Dictionary of parameters
        :return: colored_printout, plot_filename
        '''
        s = ''
        fancy_s = ''
        for key, val in d.items():
            if val is None: continue
            s += f'{key}{val}_'
            fancy_s += f'{Fore.LIGHTYELLOW_EX}{key}{Fore.RESET}=' \
                       f'{Fore.LIGHTCYAN_EX}{val}{Fore.RESET}, '
        return s[:-1], fancy_s[:-2]

    # Printout the parameters used this run
    params = {'pos': pos, 'train_st': train_st, 'train_end': train_end,
              'test_st': test_st, 'test_end': test_end,
              'best_expects_cutoff': best_experts_cutoff,
              'concat_correction': concat_correction,
              'min_rank': min_rank, 'max_rank': max_rank,
              'do_mean': do_mean,
              'do_CV_density': do_CV_density,
              'horz_smooth_amt': horz_smooth_amt,
              'vert_smooth_amt': vert_smooth_amt,
              'vert_smooth_correct': vert_smooth_correct,
              'concat_proximal_ranks': concat_proximal_ranks,
              'eval_float': eval_float,
              'density_resolution': density_resolution}
    _, colored_printout = organize_params(params)
    print(colored_printout)

    # Get a filename for files generated using these parameters
    params_short = {'pos': pos, 'trs': train_st, 'tre': train_end,
                    'tss': test_st, 'tse': test_end,
                    'bec': best_experts_cutoff, 'rbw': concat_correction,
                    'mlr': min_rank,
                    'mr': max_rank,
                    'CVd': do_CV_density, 'hsm': horz_smooth_amt,
                    'vsm': vert_smooth_amt,
                    'vsc': vert_smooth_correct,
                    'dm': do_mean, 'concat': concat_proximal_ranks,
                    'ef': eval_float, 'dres': density_resolution}
    params_fn, _ = organize_params(params_short)
    print(f'{params_fn=}')
    experts = df_rankings.columns[:-5]

    # Split into train and test
    if single_year_test:
        df_train, df_test = get_train_test_single_year(df_rankings, train_st,
                                                       test_st)
    else:
        df_train, df_test = get_train_test(df_rankings, train_st, train_end,
                                           test_st, test_end)
    print(f'train years: {df_train.year_.unique()}')
    print(f'test years: {df_test.year_.unique()}')


    # Filter experts to the best ones, then add average of all expert ranks
    if best_experts_cutoff > 0:
        fp_experts = os.path.join(PKL_CACHE,
            f'experts_{pos}_{train_st}_{train_end}_{best_experts_cutoff}.pkl')
        experts = pickle_wrap(fp_experts,
                              lambda: get_best_experts(df_train, experts,
                                                      tile=best_experts_cutoff))
    add_mean_expert(df_train, experts)
    add_mean_expert(df_test, experts)


    # Get the expert distribution (ED)
    experts_used = ['mean_expert'] if do_mean else experts
    ED = get_expert_distribution(df_train, experts_used, pos=pos,
                                 concat_proximal_ranks=concat_proximal_ranks,
                                 do_CV_density=do_CV_density,
                                 horz_smooth_amt=horz_smooth_amt,
                                 vert_smooth_amt=vert_smooth_amt,
                                 concat_correction=concat_correction,
                                 max_rank=
                                 max_rank+12 if pos in ['RB', 'WR'] else
                                (max_rank+6 if pos in ['TE', 'QB'] else
                                 max_rank),
                                 density_resolution=density_resolution,
                                 vert_smooth_correct=vert_smooth_correct,
                                 plot_name=plot_name)

    # Convert the density to a log distribution then evaluate test accuracy
    ED.convert_to_log_density()
    if not do_test:
        return ED, colored_printout, params_fn

    test_acc, test_size = get_test_accuracy(df_test, ED, min_rank=min_rank,
                                            max_rank=max_rank,
                                            eval_float=eval_float, pos=pos)
    ED.set_score(test_st, test_acc, test_size)

    # Save the expert distribution (ED) and plot
    save_ED_plot(ED, test_size, params_fn, test_acc, max_rank,
                 single_year_test, test_st)
    return test_acc, test_size


def get_expert_distribution(df_train, experts_used, pos='RB', max_rank=48,
                            concat_proximal_ranks=1,
                            do_CV_density=False, horz_smooth_amt=False,
                            concat_correction=1.0,
                            density_resolution=2500, vert_smooth_amt=2,
                            vert_smooth_correct=False,
                            plot_name=''):
    '''
    Takes the input data and creates the an Expert Distribution object. This
        object contains the density array. The density array procedures
        are described in the white paper. This function runs all the steps
        described in the white paper.

    :param df_train: Dataframe containing all the ranking and scores data
    :param experts_used: Expert names whose rankings are used. Usually just
        ['mean_expert'], with 'mean_expert' being treated as an expert.
        In principle, this code can be used with many experts, like ['tom',
        'bill'], and the resulting Expert_Distribution object will hold onto all
        of them. However, the white paper describes using only a single expert,
        'mean_expert'.
    :param pos: Position (str)
    :param max_rank: Worst ranked considered (higher rank = worse)
    :param concat_proximal_ranks: If neighboring ranks are concatenated, how
        many should be concatenated? If 2, then ranks [i-2:i+2] are concatenated
    :param do_CV_density: If True, cross-validation will be used to find the
        optimal bandwidth density for kernel density estimation? If not, it will
        use a constant, which is a default parameter in another function
    :param horz_smooth_amt: How much horizontal smoothing should be applied?
    :param concat_correction: If non-zero, a concatenation correction is applied.
        See the white paper for details on why biases arise at high ranks.
        Although this specific correction is not described in the white paper
        it is similar to the correction used for vertical smoothing, which
        was described. Concat_correction should be a float from [0.0, 1.0],
        where 0.0 = no correction and 1.0 = full correction.
    :param density_resolution: Kernel density is calculated discretely (e.g.,
        density of 0 pts, 0.1 pts, etc). density_resolution specifies the number
        of discrete points.
    :param vert_smooth_amt: Float specifies how much vertical smoothing to apply
    :param vert_smooth_correct: If True, applies the vertical smoothing
        correction described in the white paper
    :param plot_name: Will be added to the filename of the saved plot
    :return:
    '''
    x_cum_tiles = np.linspace(0, 1, density_resolution)
    if pos == 'DST':
        x_density_pts = np.linspace(-10, 40, density_resolution)
    else:
        x_density_pts = np.linspace(0, 50, density_resolution)
    ar_density_l = []
    good_experts = []
    assert len(experts_used), f'Error, no experts used: {experts_used=}'
    # This loop calculates the density distributions for each expert.
    #   However, the white paper only uses one expert, 'mean_expert'
    for expert in tqdm(experts_used, desc='Getting density_ar for each expert'):
        ar_density = get_density_array(df_train, expert, pos, x_density_pts,
                               x_cum_tiles, max_rank=max_rank,
                               num_neighbors_concat=concat_proximal_ranks,
                               do_CV_density=do_CV_density,
                               horz_smooth_amt=horz_smooth_amt,
                               concat_correction=concat_correction,
                               vert_smooth_amt=vert_smooth_amt,
                               vert_smooth_correct=vert_smooth_correct)
        ar_density = ar_density[:max_rank, :]
        ar_density_l.append(ar_density)
        good_experts.append(expert)
    ar_density_all = np.array(ar_density_l)
    year_end = df_train.index.get_level_values(1).max()
    year_st = df_train.index.get_level_values(1).min()
    ED = Expert_Distribution(good_experts, ar_density_all, x_density_pts,
                             x_cum_tiles,
                             pos, max_rank, year_st, year_end,
                             plot_name=plot_name)
    return ED


def cross_val_years(df_rankings, st_year, params):
    '''
    Runs fit_and_predict_testing(...) with cross-validation. Cross-validation
        is done by organizing each year's data into a fold. For example,
        fitting on years 2018, 2019, 2021, and predicting on 2020.

    The ED saved will be fit using all the years and without accuracy tested

    :param df_rankings: Input data
    :param st_year: Start year
    :param params: Params to be passed to fit_and_predict_testing(...)
    :return: total_accuracy (accuracy, log likelihood)
    :return: total_size (total size of all test sets; number of examples)
    '''
    scores = []
    cnts = []
    df_rankings = df_rankings[df_rankings.index.get_level_values(1) >= st_year]

    max_yr = df_rankings.index.get_level_values(1).max()

    params['train_st'] = st_year
    params['train_end'] = None
    params['test_end'] = None
    params['single_year_test'] = True
    for test_year in range(st_year+1, max_yr+1):
        print(f'\tTest year: {test_year}')
        params['test_st'] = test_year
        score_yr, cnt = fit_and_predict_testing(deepcopy(df_rankings), **params)
        scores.append(score_yr)
        cnts.append(cnt)
    total_accuracy = sum(scores)
    print(f'Total score: {total_accuracy:.3f}')
    print(f'^{params}')
    total_size = sum(cnts)
    params['train_st'] = st_year
    params['train_end'] = max_yr + 1
    params['single_year_test'] = False
    ED, colored_printout, params_fn = fit_and_predict_testing(df_rankings,
                                                              **params)
    out_dir = r'\Weekly_Distributions\full_EDs'

    fn = f'acc{total_accuracy:.3f}_cnt{total_size}_{params_fn}'
    fn_dir = f'{params["pos"]}_max_{params["max_rank"]}'
    Path(os.path.join(out_dir, 'figs', fn_dir)).mkdir(parents=True,
                                                      exist_ok=True)
    Path(os.path.join(out_dir, 'EDs', fn_dir)).mkdir(parents=True,
                                                     exist_ok=True)

    ED.set_fn(os.path.join(fn_dir, fn))
    ED.set_out_dir(out_dir)
    ED.set_score('CV', total_accuracy, total_size)
    ED.set_scores_by_year(list(range(st_year,
                df_rankings.index.get_level_values(1).max()+1)), scores, cnts)
    ED.plot_density(save_plot=True)
    ED.plot_density(save_plot=True, cumdens=True)
    ED.set_params_full(colored_printout)
    ED.save()
    return total_accuracy, total_size


def run_params(df_rankings, params, CV=True, no_test=False):
    '''
    Runs fit_and_predict_testing(...) for a given set of parameters

    :param df_rankings: Input data, see load_data(...)
    :param params: Parameters to be passed to fit_and_predict_testing(...)
    :param CV: If True, fit_and_predict_testing(...) is tested with
        cross-validation
    :param no_test: If True, then testing accuracy is not tested
    :return:
    '''
    key = tuple(params.items())
    if CV:
        score, cnt = cross_val_years(df_rankings, 2016, params)
    else:
        if no_test:
            score, cnt = fit_and_predict_testing(df_rankings,
                                                 train_st=YEAR_ST,
                                                 train_end=YEAR_END,
                                                 test_st=YEAR_END-1,
                                                 test_end=YEAR_END,
                                                 **params)
        else:
            score, cnt = fit_and_predict_testing(df_rankings,
                                                 train_st=YEAR_ST,
                                                 train_end=YEAR_END-1,
                                                 test_st=YEAR_END-1,
                                                 test_end=YEAR_END,
                                                 **params)
    return key, score, cnt


def get_final_params(pos):
    '''
    Contains the final parameters used to create the final distributions in the
        white paper. These parameters maximize testing accuracy. Different
        parameters were used for each position.

    :param pos: Position (str)
    :return:
    '''
    if pos == 'WR':
        params = {'best_experts_cutoff': 0,
              'concat_correction': 1,
              'year_boost_weight': 0,
              'horz_smooth_amt': 60,
              'do_CV_density': True,
              'concat_proximal_ranks': 5,
              'max_rank': 90,
              'pos': 'WR',
              'eval_float': False,
              'vert_smooth_amt': 6,
              'vert_smooth_correct': True,
              'plot_name': ''}
    elif pos == 'RB':
        params = {'best_experts_cutoff': 0,
              'concat_correction': 1,
              'horz_smooth_amt': 60,
              'do_CV_density': True,
              'concat_proximal_ranks': 5,
              'max_rank': 72,
              'pos': 'RB',
              'eval_float': False,
              'vert_smooth_amt': 8,
              'vert_smooth_correct': True,
              'plot_name': ''}
    elif pos == 'QB':
        params = {'best_experts_cutoff': 0,
              'concat_correction': 0,
              'horz_smooth_amt': 60,
              'do_CV_density': True,
              'concat_proximal_ranks': 6,
              'max_rank': 30,
              'pos': 'QB',
              'eval_float': False,
              'vert_smooth_amt': 2,
              'vert_smooth_correct': True,
              'plot_name': ''}
    elif pos == 'TE':
        params = {'best_experts_cutoff': 0,
              'concat_correction': 1,
              'horz_smooth_amt': 60,
              'do_CV_density': True,
              'concat_proximal_ranks': 2,
              'max_rank': 30,
              'pos': 'TE',
              'eval_float': False,
              'vert_smooth_amt': 6,
              'vert_smooth_correct': True,
              'plot_name': ''}
    elif pos == 'DST':
        params = {'best_experts_cutoff': 0,
              'concat_correction': 0.5,
              'horz_smooth_amt': 60,
              'do_CV_density': True,
              'concat_proximal_ranks': 9,
              'max_rank': 30,
              'pos': 'DST',
              'eval_float': False,
              'vert_smooth_amt': 4,
              'vert_smooth_correct': True,
              'plot_name': ''}
    else:
        raise ValueError(f'Invalid pos: {pos}')
    return params


if __name__ == '__main__':
    YEAR_ST, YEAR_END = 2013, 2022
    df = load_data(year_st=YEAR_ST, year_end=YEAR_END)

    for pos in ['DST']:
        PARAMS = get_final_params(pos)
        run_params(deepcopy(df), PARAMS, CV=False, no_test=True)


