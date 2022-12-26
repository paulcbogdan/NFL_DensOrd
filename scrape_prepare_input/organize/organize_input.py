import warnings

import pandas as pd
import numpy as np

from scrape_prepare_input.scrape.scrape_fantasypros import \
    get_all_seasons_weekly_ranks

from scrape_prepare_input.organize.fix_names import fix_fantasypros_name, \
    fix_scores_name


def get_df_rankings_scores(year_st, year_end):
    '''
    Loads the primary dataset. The dataset is made by loading and combining two:
        df_rankings: This represents all the expert ranking data.
            It contains every ranking every expert has given to every player
            for every week for every year.
        df_scores: It represents the actual scores of every player every week.
            It also contains data on the player's team and the player's FanDuel
            salary, although these are not presently used

    df_rankings is scraped from FantasyPros.com.
        See get_all_seasons_weekly_ranks(...)
    df_scores is just a .csv downloaded from a website

    The output is a dataframe with a multiindex:
        ('pos' [position], 'year', 'week', 'player' [player name])

    An annoying feature of these datasets is that some players have different
        names across datasets.
        e.g., 'Benny Snell' in df_scores is 'Benny Snell Jr.' in df_rankings
        When multiple players have the same name, this also may create problems
        fix_names.py tries to fix these issues. It uses some basic strategies,
        like testing whether adding 'jr' leads to a match. However, there is
        also some things that I hardcoded. For obscure players that are ranked
        extremely badly, and who wouldn't make there way into the model anyway,
        I didn't bother
    :param year_st: Start year
    :param year_end: End year
    :return: The resulting dataset containing player's weekly expert rankings
        and their historic weekly performances (points scored)
    '''
    df_scores = get_df_scores()
    df_scores.set_index(['pos', 'year', 'week', 'player'], inplace=True)

    df_rankings = get_all_seasons_weekly_ranks(year_st, year_end, n_inst=10,
                                               override=False)
    df_rankings['player'] = df_rankings['player'].apply(fix_fantasypros_name)
    df_rankings, all_experts = \
        pivot_experts2columns(df_rankings,
                              groupby_code=('year', 'week', 'pos', 'player'))
    df_rankings.set_index(['pos', 'year', 'week', 'player'], inplace=True)

    names_in_both = set(df_rankings.index).intersection(set(df_scores.index))

    df_rankings = df_rankings.loc[names_in_both]
    df_scores = df_scores.loc[names_in_both]
    scores_columns = ['points', 'salary', 'Team']
    df_rankings.loc[names_in_both, scores_columns] = \
        df_scores.loc[names_in_both][scores_columns]

    df_rankings.sort_index(level=[1, 2], inplace=True)
    df_rankings['year_'] = df_rankings.index.get_level_values(1)
    df_rankings['week_'] = df_rankings.index.get_level_values(2)
    return df_rankings, df_scores


def get_df_scores():
    df_scores = pd.read_csv(r'input_data/Points_and_FD_Salaries/scores_Aug20_22.csv')
    df_scores['player'] = df_scores.apply(fix_scores_name, axis=1)
    df_scores['points'] = df_scores['Points'].astype(float)
    df_scores['salary'] = df_scores['Salary'].apply(
        lambda x: int(x[1:].replace(',', '')) if \
        (isinstance(x, str) and x != 'N/A') else np.nan)
    df_scores.drop(['Points', 'Name', 'Salary'], axis=1, inplace=True)
    return df_scores


def pivot_experts2columns(df_rankings, groupby_code=('year', 'pos', 'Player')):
    '''
    Takes df_rankings, where each expert's ranking is represented by its own row
        and pivots so each expert's ranking is represented by its own column

    If an expert didn't complete any rankings for a given year/week, their ranks
        are set to NaN

    If an expert ranked some players but not all, the missing rankings are
        filled based on the average ranking of other experts

    This function can operate with either weekly rankings (full_year=False) or
        yearly draft rankings (full_year=True).

    The code also does some checks to make sure the data are organized properly

    Note: 'ECR' = Expert Consensus Ranking, which is a column from FantasyPros

    :param df_rankings:
    :param groupby_code: Specifies the categories ranked
    :return:
    '''
    df_dr_grouped = df_rankings.groupby(list(groupby_code))
    df_as_l = []
    all_experts = set()
    for tup, df_dr_group in df_dr_grouped:
        experts = df_dr_group['Expert']
        all_experts.update(experts)
        ranks = df_dr_group['Rank']
        ECR = df_dr_group['Rank'] + df_dr_group['vs_ECR']
        ECR.dropna(inplace=True)
        if len(ECR.unique()) > 1:
            df_dr_group['sum'] = df_dr_group['Rank'] + df_dr_group['vs_ECR']
            if tup == (2022, 'TE', 'Taysom Hill'):
                ECR = 43
                # Taysom Hill in 2022 is a special case
                # He's the only player to have ever been both a TE and a QB
            else:
                warnings.warn(f'Too many ECR: [{tup}] {len(ECR.unique())=} | '
                              f'{ECR.unique()}')
                ECR = ECR.unique()[0]
        elif len(ECR.unique()) == 0:
            ECR = np.nan
        else:
            ECR = ECR.unique()[0]

        experts_no_ECR = [f'{expert}_no_ECR' for expert in experts]
        new_row = dict(zip(list(experts.values) + list(experts_no_ECR) + ['ECR', 'ECR_no_ECR'],
                           list(ranks.values)  + list(ranks.values) + [ECR, ECR]))
        new_row.update(dict((code, tup_val) for
                            code, tup_val in zip(groupby_code, tup)))
        new_row.update({'pos': df_dr_group['pos'].iloc[0]})
        df_as_l.append(new_row)

    return pd.DataFrame(df_as_l), list(all_experts) + ['ECR']



