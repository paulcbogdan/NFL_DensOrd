import string

'''
This code is used to fix the names of players in the df_rankings and df_scores
    dataframes. This code is not commented
'''

THREE_PART_NAMES = {'austin-seferian-jenkins', 'ted-ginn-jr', 'zach-miller-chi', 'damien-harris-rb', 'matt-jones-rb',
                    'kansas-city-defense', 'mo-alie-cox', 'zach-miller-oak', 'michael-thomas-wr', 'new-england-defense',
                    'josh-robinson-rb', 'damon-sheehy-guiseppi', 'josh-allen-qb', 'tampa-bay-defense', 'irv-smith-jr',
                    'donovan-peoples-jones', 'antonio-gandy-golden', 'darrius-heyward-bey', 'clyde-edwards-helaire',
                    'green-bay-defense', 'stevie-scott-iii', 'ronald-jones-ii', 'david-johnson-rb', 'mike-williams-tb',
                    'los-angeles-defense', 'san-francisco-defense', 'san-diego-defense', 'las-vegas-defense',
                    'maurice-jones-drew', 'mike-davis-rb', 'ray-ray-mccloud', 'justin-jackson-rb', 'bryce-love-rb',
                    'josh-hill-te', 'benjamin-snell-jr', 'juju-smith-schuster', 'adrian-peterson-min',
                    'larod-stephens-howling', 'mike-williams-wr', 'equanimeous-st-brown', 'dj-moore-wr',
                    'michael-carter-rb', 'stanley-morgan-jr', 'mike-thomas-wr', 'ryan-grant-wr', 'new-orleans-defense',
                    'elijah-mitchell-rb', 'devin-smith-wr', 'benjarvus-green-ellis', 'charles-johnson-wr',
                    'otis-anderson-jr', 'dorial-green-beckham', 'ricky-seals-jones', 'jj-arcega-whiteside',
                    'cameron-artis-payne', 'robert-griffin-iii', 'chris-herndon-iv', 'dalton-keene-te',
                    'najee-harris-rb', 'patrick-mahomes-ii',
                    'tony-jones-rb', 'benny-snell-jr', 'lynn-bowden-jr'}
GOOD_THREE_PARTs = {'maurice-jones-drew', 'darrius-heyward-bey',  'benjarvus-green-ellis', 'larod-stephens-howling',
                    'cameron-artis-payne', 'austin-seferian-jenkins', 'ted-ginn-jr', 'mo-alie-cox', 'irv-smith-jr',
                    'clyde-edwards-helaire', 'larod-stephens-howling', 'ray-ray-mccloud', 'benjamin-snell-jr',
                    'juju-smith-schuster', 'dorial-green-beckham', 'ricky-seals-jones', 'jj-arcega-whiteside',
                    'robert-griffin-iii', 'chris-herndon-iv', 'patrick-mahomes-ii', 'amon-ra-st-brown'}
EXCL = {'zach-miller-oak'}
EXCL_DUPLICATE = {'mike_thomas'}
MANUAL_MAPPING = {'Jeffery Wilson': 'Jeff Wilson Jr.', 'Christopher Ivory': 'Chris Ivory',
                  'Amonra Stbrown': 'Amon-Ra St. Brown', 'Benjamin Snell Jr': 'Benny Snell Jr.',
                  'Benjamin Watson': 'Ben Watson', 'Chris Herndon Iv': 'Chris Herndon'}
DF_SCORES_MAPPING = {'Patrick Mahomes II': 'Patrick Mahomes',
                     'Mitchell Trubisky': 'Mitch Trubisky',
                     'Kenny Gainwell': 'Kenneth Gainwell',
                     'Rob Kelley': 'Robert Kelley',
                     'Benny Snell': 'Benny Snell Jr.'}
DST_DF_RANKINGS_MAP = {'pittsburgh_defense': 'PIT', 'cleveland_defense': 'CLE', 'jacksonville_defense': 'JAC',
                       'detroit_defense': 'DET', 'miami_defense': 'MIA', 'carolina_defense': 'CAR',
                       'cincinnati_defense': 'CIN',
                       'los_angeles_defense': 'STL', 'denver_defense': 'DEN', 'chicago_defense': 'CHI',
                       'new_england_defense': 'NE', 'las_vegas_defense': 'LV', 'philadelphia_defense': 'PHI',
                       'buffalo_defense': 'BUF', 'tampa_bay_defense': 'TB', 'green_bay_defense': 'GB',
                       'dallas_defense': 'DAL',
                       'indianapolis_defense': 'IND', 'san_diego_defense': 'SD', 'washington_defense': 'WAS',
                       'houston_defense': 'HOU', 'san_francisco_defense': 'SF', 'atlanta_defense': 'ATL',
                       'seattle_defense': 'SEA', 'arizona_defense': 'ARI', 'baltimore_defense': 'BAL',
                       'kansas_city_defense': 'KC', 'new_orleans_defense': 'NO', 'tennessee_defense': 'TEN',
                       'minnesota_defense': 'MIN'}
DF_SCORES_DST = ['DAL', 'KC', 'MIA', 'BUF', 'TEN', 'PHI', 'WAS', 'DET', 'DEN',
                 'TB', 'NYJ', 'ARI', 'HOU', 'SEA', 'STL', 'NO',
        'CAR', 'CHI', 'CLE', 'IND', 'JAC', 'SF', 'NE', 'LV', 'ATL', 'PIT',
                 'MIN', 'NYG', 'SD', 'BAL', 'CIN', 'GB']


DEF_NAME_MAPPER = {'jac': 'JAC', 'bal': 'BAL', 'phi': 'PHI', 'det': 'DET', 'car': 'CAR', 'dal': 'DAL',
                   'was': 'WAS', 'buf': 'BUF', 'gnb': 'GB', 'ari': 'ARI', 'lac': 'LAC', 'sea': 'SEA',
                   'den': 'DEN', 'atl': 'ATL', 'cin': 'CIN', 'cle': 'CLE', 'sfo': 'SF', 'kan': 'KC',
                   'oak': 'LV', 'chi': 'CHI', 'min': 'MIN', 'nyg': 'NYG', 'ten': 'TEN', 'ind': 'IND',
                   'nwe': 'NE', 'nor': 'NO', 'hou': 'HOU',
                   'mia': 'MIA', 'tam': 'TB', 'nyj': 'NYJ', 'stl': 'STL', 'pit': 'PIT', 'sdg': 'SD',
                   'lar': 'LAR', 'lvr': 'OAK'}


def fix_fantasypros_name(name):
    if name in EXCL_DUPLICATE:
        return 'BAD'
    if name in DST_DF_RANKINGS_MAP:
        return DST_DF_RANKINGS_MAP[name]
    name_dash = name.replace('_', '-')
    if name_dash in THREE_PART_NAMES:
        if name_dash not in GOOD_THREE_PARTs and name_dash not in EXCL:
            name = '_'.join(name.split('_')[:2])
    return name.replace('_', ' ').title()


def fix_scores_name(row):
    if row['pos'] == 'DST':
        name_reformatted = DEF_NAME_MAPPER[row['Team']]
    else:
        name = row['Name']
        name_spl = name.split(', ')
        name_reformatted = name_spl[1] + ' ' + name_spl[0]
    return name_reformatted


def update_name(name, lower_to_true_name):
    clean_name = name.lower()
    clean_name = clean_name.translate(str.maketrans('', '', string.punctuation))
    if clean_name not in lower_to_true_name:
        return name
    else:
        return lower_to_true_name[clean_name]


def update_extensions(name, df_source_name_set):
    if name in MANUAL_MAPPING:
        return MANUAL_MAPPING[name]
    if name in df_source_name_set:
        return name
    for extension in ['Jr.', 'II', 'III', 'IV', 'V']:
        if f'{name.title()} {extension}' in df_source_name_set:
            return f'{name.title()} {extension}'
        else:
            sans_extension = name.replace(f' {extension}', '').title()
            if sans_extension in df_source_name_set:
                return sans_extension
    return name


def align_names(df_source, df_fix, name_key='Player'):
    df_fix_set = set(df_fix[name_key])
    print(list(df_source.columns))
    df_source_name_set = set(df_source[name_key])
    print(f'Pre align: source x fix: {len(df_fix_set.intersection(df_source_name_set))/len(df_fix_set)}')
    lower_to_true_name = dict((name.lower().translate(str.maketrans('', '', string.punctuation)), name) for name in df_source_name_set)
    lower_to_true_name.update(dict((name.lower().translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))), name) for name in df_source_name_set)) # covers juju smith schuster, and maybe others?
    df_fix[name_key] = df_fix[name_key].apply(update_name, args=(lower_to_true_name, ))
    df_fix[name_key] = df_fix[name_key].apply(update_extensions, args=(df_source_name_set, ))
    df_fix_set = set(df_fix[name_key])
    print(f'Post align: source x fix: {len(df_fix_set.intersection(df_source_name_set)) / len(df_fix_set)}')