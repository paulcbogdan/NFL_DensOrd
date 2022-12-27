'''
This file contains some basic operations related to pandas dataframes
'''

def get_train_test(df, train_st, train_end, test_st, test_end):
    df_train = get_df_years(df, train_st, train_end)
    df_test = get_df_years(df, test_st, test_end)
    return df_train, df_test

def get_df_pos(df, pos):
    return df[df.index.get_level_values(0) == pos]

def get_df_years(df, year_st, year_end):
    return df[(df.index.get_level_values(1) >= year_st) & (df.index.get_level_values(1) < year_end)]

def get_train_test_single_year(df, train_st, test_yr):
    df_train = df[(df.index.get_level_values(1) >= train_st) & (df.index.get_level_values(1) != test_yr)]
    df_test = df[(df.index.get_level_values(1) == test_yr)]
    return df_train, df_test