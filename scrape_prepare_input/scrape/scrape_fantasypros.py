import os
import re
import time
from functools import partial

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from make_distributions.utils import PKL_CACHE, pickle_wrap


def get_all_seasons_weekly_ranks(year_st=2018, year_end=2021, n_parallel=5,
                                 override=False):
    '''
    Scrapes all weekly rankings from FantasyPros.com for all players from years
        year_st to year_end. Scrapes all positions. Scraping is structured as a
        three-level hierarchy
    1. get_all_seasons_weekly_ranks() function iterates through all years.
        For each year, it calls get_full_season_ranks(year),
    2. get_full_season_ranks(year) iterates over all players.
        For each player, get_full_season_ranks(year) calls
        get_player_season_weekly_ranks(...)
    3. get_player_season_weekly_ranks(...) scrapes a full season for
        a given player.
    This hierarchy is convenient for saving intermittently (each year), as
        scraping everything will take a while. Additionally, it is convenient
        for parallelization, as players are scraped in parallel.
    If override=False, it will use existing year's .pkls files if they exist.
        You may be interested in this if you processed 2013-2021 already and
        just want to add 2022. If override=True, it will redo all of them
    The data are organized as a dataframe where each row is one expert's ranking
        of one player for one week.
        For example, Rich Piazza ranked tom_brady as QB #10 in week 2 of 2020

    All data scraped is fully public and freely visible to website visitors.
        In total roughly 64,000 webpages are scraped, containing over a million
        expert ranks (10 years x 16-17 weeks x ~400 players x 50-300 experts).
        FantasyPros.com will inevitably time out or stop servicing requests.
        The code below handles this, takes a 5-minute sleep, and reattempts if
        FantasyPros.com refuses to respond.

    :param year_st: Start year
    :param year_end: End year
    :param n_parallel: Number of instances for parallelization
    :param override: Whether to override existing data
    :return: df_rankings, one row per expert per player per week per year
    '''
    df_rankings = []
    for year in range(year_st, year_end):
        fp_pkl = os.path.join(PKL_CACHE, f'df_ranks_yr{year}.pkl')
        df_year, _ = pickle_wrap(fp_pkl,
                                 lambda: get_full_season_ranks(year,
                                                        n_parallel=n_parallel),
                                 easy_override=override)
        df_rankings.append(df_year)
    return pd.concat(df_rankings)


def get_full_season_ranks(year, n_parallel=5):
    '''
    Scrapes all weekly rankings for all players (all positions) for a given year
        See get_all_seasons_weekly_ranks(...) for more info.
    There is a lot of data to scrape:
        10 years x 16-17 weeks x ~400 players x 50-300 experts.
        Parallelization helps make this go by faster. n_parallel controls this.

    Each player's data is contained on its own unique webpage, so the first
        thing this function does is get all the webpages it needs via
        get_names_php(year). See below.

    :param year: single year to scrape
    :param n_parallel: Number of instances for parallelization
    :return: df_rankings, one row per expert per player per week
    '''
    php_players, _, pos_l = get_names_php(year)
    print(f'get_all_players({year}, n_inst={n_parallel}): {len(php_players)=}')
    start = time.time()
    get_player_season_year = partial(get_player_season_weekly_ranks, year=year)
    df_all = process_map(get_player_season_year, php_players, pos_l,
                         max_workers=n_parallel)
    df_all = [df for df_season in df_all for df in df_season]
    print(f'{year}: Time needed for {len(php_players)} players, '
          f'concurrent ({n_parallel} instances): {time.time() - start:.3f}s')
    return pd.concat(df_all), None


def get_names_php(year=2019, week=None, sleep_time=300):
    '''
    Gets lists of name_php for all players ranked by FantasyPros in a given year
        For example, ['tom-brady.php', 'drew-brees.php', ...]. Returns a single
        list for players of all positions
    Other functions loop through the names to create each URL to be scraped
        See scrape_FantasyPros_player(...)
        base_url = f'https://www.fantasypros.com/nfl/rankings/{name_php}.php'
    Also, returns a list of positions for each player (e.g. ['QB', 'QB', ...])
        Indices match up with the name_php list

    name_php with 2+ dashes in their name (e.g., clyde-edwards-helaire.php)
        are sometimes helpful to have, and a list of just those are returned.

    :param year: Year to get player names for
    :param sleep_time: If scrape fails, how long to sleep before trying again
    :return: names_php_all_pos (all name_php for all players of all positions)
         pos_l (list of positions for each player)
         names_php_many_dash_all_pos (all name_php for all players of all
            positions, who have more than one dash in their name)
    '''

    names_php_all_pos = []
    names_php_many_dash_all_pos = []
    pos_l = []
    for pos in ['qb', 'rb', 'wr', 'te', 'dst']:
        if week is None:
            base_url = f'https://www.fantasypros.com/nfl/rankings/' \
                       f'{pos}-cheatsheets.php'
            r = requests.get(base_url, params={'year': year})
        else:
            base_url = f'https://www.fantasypros.com/nfl/rankings/{pos}.php'
            r = requests.get(base_url, params={'year': year, 'week': week})

        soup = BeautifulSoup(r.text, features='lxml')
        for i, content in enumerate(soup.find_all('script',
                                                  {'type': 'text/javascript'})):
            if 'adpData' in str(content):
                break
        else:
            print(f'Failure to scrape player names for {year=}')
            print(f'\tWaiting {sleep_time} seconds and trying again.')
            return get_names_php(year, sleep_time)

        names_php_one_dash = re.findall('"\w*-\w*.php"', str(content))
        names_php_one_dash = [php_player.replace('"', '').replace('.php', '')
                       for php_player in names_php_one_dash]

        names_php_many_dash = re.findall('"\w*-\w*-\w*.php"', str(content))
        names_php_many_dash += re.findall('"\w*-\w*-\w*-\w*.php"',
                                            str(content))
        names_php_many_dash += re.findall('"\w*-\w*-\w*-\w*-\w*.php"',
                                            str(content))

        names_php_many_dash = [name_php.replace('"', '').replace('.php', '')
                                 for name_php in names_php_many_dash]

        names_php = list(set(names_php_one_dash + names_php_many_dash))
        names_php_all_pos += names_php
        pos_l += [pos] * len(names_php)
        names_php_many_dash_all_pos += list(set(names_php_many_dash))

    return names_php_all_pos, pos_l, names_php_many_dash_all_pos


def get_player_season_weekly_ranks(name_php, pos, year=2019):
    '''
    Scrapes all weekly rankings for a given player for a given year
        See get_all_seasons_weekly_ranks(...) for more info

    :name_php: php name of player to scrape (e.g., 'tom-brady.php')
    :param pos': position of player to scrape ('QB', 'RB', 'WR', 'TE', 'DST')
    :return: df_rankings, one row per expert per week
    '''
    df_player_weeks = []
    for week in tqdm(range(1, 17), desc='get_player_season'):
        df_player_week, _ = scrape_FantasyPros_player(name_php, year, pos,
                                                      weekly=True, week=week)
        df_player_weeks.append(df_player_week)
    print(f'Done get_player_season: {name_php}, {len(df_player_weeks)=}')
    return df_player_weeks


def scrape_FantasyPros_player(name_php, year, pos, weekly=False, week=None,
                              attempts_remaining=5, sleep_time=300):
    '''
    Scrapes ranking data for a single player from FantasyPros.com.
        Example URL: https://www.fantasypros.com/nfl/rankings/tom-brady.php
        Scrapes either weekly rankings and pre-season (yearly) draft rankings

    :param name_php: # Name of player for URL, like 'tom-brady.php'
    :param year: Year
    :param pos: Position of player. Used to specify URL parameters
    :param weekly: If True, gets weekly ranks. If False, gets draft (year) ranks
    :param week: If weekly, which week to get
    :param attempts_remaining: If scrape fails, how many more times to try
    :param sleep_time: If scrape fails, how long to sleep before trying again
    :return: Returns dataframe, representing how all experts ranked a player
        in a week (if weekly=True) or in given year (if weekly=False).
        Columns include: 'Expert', 'Affiliation', 'Rank', 'vs_ECR',
                         'Expert_2021_Accuracy'
    '''

    def get_rank_from_rank_str(rank_str):
        '''
        :param rank_str: FantasyPros represents non-null ranks as strings: '*#{int}'
            such as 'RB #1', 'WR #20', or plainly '#200'.
            Null ranks are represented as '-' or 'NR'.
        :return: int (if non-null rank) or pd.NA (if null rank)
        '''
        if rank_str in ['-', 'NR']:
            return pd.NA  # Note: pd.NA works better in a column of ints than np.nan
        assert isinstance(rank_str, str), \
            f'pos_rank not a str: {rank_str=} ({type(rank_str)})'
        assert '#' in rank_str, f'# not in pos_rank: {rank_str=}'
        return int(rank_str.split('#')[1])

    def clean_vs_ECR_string(vs_ECR):
        '''
        :param vs_ECR: FantasyPros represents positive vs_ECR as strings: '+{int}'.
            Negative vs_ECR are represented as '-{int}'.
            Null ranks are represented as '-' or 'NR'.
        :return: int (if non-null vs_ECR) or pd.NA (if null vs_ECR)
        '''
        if vs_ECR in ['-', 'NR']: return pd.NA
        vs_ECR = vs_ECR.replace('+', '')  # Removes plus sign, if present
        return int(vs_ECR)

    def get_pos_from_rank_str(rank_str):
        '''
        Above format_fantasypros_rank(rank_str) extracts the ranking from  rank_str
            e.g., extracts 1 from 'RB #1'
        This function extracts the position from rank_str
            e.g., extracts 'RB' from 'RB #1'

        :param rank_str: FantasyPros represents non-null ranks as strings: '*#{int}'
            such as 'RB #1', 'WR #20', or plainly '#200'.
            Null ranks are represented as '-' or 'NR'.
        :return: int (if non-null rank) or pd.NA (if null rank)
        '''
        if rank_str in ['-', 'NR']: return pd.NA
        assert isinstance(rank_str, str), \
            f'rank_str not a str: {rank_str=} ({type(rank_str)})'
        assert '#' in rank_str, f'# not in pos_rank: {rank_str=}'
        pos = rank_str.split('#')[0].replace(' ', '')
        assert len(pos) > 0, f'No position: {rank_str=}'
        return pos

    base_url = f'https://www.fantasypros.com/nfl/rankings/{name_php}.php'
    if weekly:
        params = {'type': 'weekly', 'year': year, 'week': week}
    else:
        params = {'type': 'draft', "year": year}
    if pos.lower() != 'qb' and pos.lower() != 'dst':
        params['scoring'] = 'HALF' # Half PPR specified for non-QBs and non-DSTs

    # Attempt to scrape the page
    r = requests.get(base_url, params=params)
    html = r.text
    soup = BeautifulSoup(html, features='lxml')
    try:
        team_element = soup.find_all('h2')[1]
    except IndexError as e:
        print(f'Failure to scrape: {name_php=} | {e=}')
        if attempts_remaining > 0:
            print(f'\tWaiting {sleep_time} seconds and trying again. '
                  f'({attempts_remaining - 1} attempts remaining)')
            time.sleep(sleep_time)
            return scrape_FantasyPros_player(name_php, year, pos, weekly,
                                             week, attempts_remaining - 1,
                                             sleep_time)
        else:
            print(f'\tNo attempts remaining. Returning empty DataFrame.')
            return pd.DataFrame(), None

    # Create dataframe from page html
    team = str(team_element).split(' ')[2]
    table = soup.find('table',
                      {'class': "table table-bordered expert-ranks sort-table"})
    if table is None:
        print(f'Can\'t process: {name_php}')
        print(f'\t{base_url=}')
        print(f'\t{params=}')
        return pd.DataFrame(), team

    df_as_list_of_lists = []
    for row_i, row in enumerate(table.find_all('tr')):
        tds = row.find_all('td')
        print(f'{tds=}')
        if row_i == 0: # Sanity check that the website structure hasn't changed
            assert len(tds) == 0, f'0th row should have [] tds, but is: {row=}'
            continue
        df_as_list_of_lists.append([val.get_text() for
                                    val in row.find_all('td')])
    df = pd.DataFrame(df_as_list_of_lists,
                      columns=['Expert', 'Affiliation', 'Rank',
                               'vs_ECR', 'Expert_2021_Accuracy'])

    # Clean up the dataframe and return
    df['pos'] = df['Rank'].apply(get_pos_from_rank_str)
    df[['Rank', 'Expert_2021_Accuracy']] = \
        df[['Rank', 'Expert_2021_Accuracy']].applymap(get_rank_from_rank_str)
    df['vs_ECR'] = df['vs_ECR'].apply(clean_vs_ECR_string)
    df['player'] = name_php.replace('-', '_')
    df['year'] = int(year)
    positions_found = [pos_ for pos_ in df['pos'].unique() if not pd.isna(pos_)]
    if weekly: df['week'] = week
    if not len(positions_found):
        df['pos'] = 'NR' # Position unknown for players who no expert ranked
    else:
        # Players who were ranked by some experts but not all have no position
        #   in some of their rows. We'll fill in the position from other rows
        df['pos'].fillna(positions_found[0], inplace=True)
        assert (df['pos'] == df['pos'].iloc[0]).all(), \
            f'Not all the elements are the same! {df.pos.unique()=}'
            # The position should be the same for all rows

    return df, team