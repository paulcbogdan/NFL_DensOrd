import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm
from datetime import date


def scrape_all_salaries(year_st=2011, year_end=2022):
    '''
    Scrapes all daily fantasy salaries for all players from year_st to year_end
        The data are organized as a dataframe where each row is one player's
        salary for a given week of a given year.

    The dataframe is saved as a .csv

    '''
    all_dfs = []
    for year in tqdm(range(year_st, year_end)):
        for week in range(1, 19):
            # in 2021, the NFL season was extended to week 18 (one more week)
            if week == 18 and year < 2021:
                continue
            df = get_week_salaries(year, week)
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    dir_out = 'Points_and_FD_Salaries'
    fp_out = os.path.join(dir_out,
                          f'scores_{date.today().strftime("%b%d_%y")}.csv')
    df.to_csv(fp_out, index=False)

def get_week_salaries(year, week, platform='fd'):
    '''
    Retrieves historic data on player's Daily Fantasy salaries. Returns a
        dataframe containing the data for every player.
        The data are organized as a dataframe where each row is one player's
        salary for a given week of a given year.

    :param year: Year (season)
    :param week: Week of the season
    :param platform: Daily Fantasy platform. 'fd' corresponds to FanDuel
    :return:
    '''
    position_mapping = {'Quarterbacks': 'QB',
                        'Running Backs': 'RB',
                        'Tight Ends': 'TE',
                        'Wide Receivers': 'WR',
                        'Kickers': 'K',
                        'Defenses': 'DST'}

    r = requests.get(
        "http://rotoguru1.com/cgi-bin/fyday.pl",
        params={
            "game": platform,
            'week': week,
            "year": year,
        },
    )
    if r.text == "Invalid date":
        raise ValueError("Invalid date provided")
    html = r.text
    soup = BeautifulSoup(html, features='lxml')
    table = soup.find("table", {"cellspacing": 5})
    table_rows = table.find_all("tr")
    pruned_rows = []
    pos = None
    for tr in table_rows:
        td = tr.find_all("td")
        row = [tr.text.strip() for tr in td]
        if 'Jump to' in row[0]:
            continue
        if row[1] == 'Team':
            pos_full = row[0]
            pos = position_mapping[pos_full]
            continue
        if not len(row[1]):
            continue
        if pos is None:
            raise('Error! no pos set!')
        row.append(pos)
        pruned_rows.append(row)
    df = pd.DataFrame( pruned_rows, columns=['Name', 'Team', 'Opponent',
                                             'Points', 'Salary', 'pos'])
    df['year'] = year
    df['week'] = week
    return df


if __name__ == '__main__':
    scrape_all_salaries()
