import numpy as np
from time import time
import pickle
import os
from random import random, shuffle
import pandas as pd
import ntpath
from colorama import Fore
import datasize
import sys

'''
Not commented
'''

PKL_CACHE = r'pickle_wrap_cache'

def setup_pd():
    pd.set_option('display.min_rows', 200)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 200)



def transform_exog_to_model(fit, exog):
    '''
    From StackOverflow
    '''
    transform=True
    self=fit
    # The following is lifted straight from statsmodels.base.model.Results.predict()
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        exog = dmatrix(self.model.data.design_info.builder, # removed .orig_exog
                       exog)
    if exog is not None:
        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]
    return exog

class Load_Timer():
    def __init__(self, name=None):
        self.start = time()
        if name == None:
            self.name = 'T'
        else:
            self.name = str(name) + ' t'

    def print_time(self, round_to=3):
        print(self.name + 'ime:', round(time()-self.start, round_to), 's')

def time_wrap(func, title=None):
    if title is None:
        lt = Load_Timer(str(func))
    else:
        lt = Load_Timer(title)
    return_content = func()
    lt.print_time()
    return return_content

def get_Bs_str(Bs):
    if Bs > 1e9:
        return f'{Fore.LIGHTMAGENTA_EX}{Bs / 1e9:.1f} GB{Fore.RESET}'
    elif Bs > 1e6:
        return f'{Fore.LIGHTRED_EX}{Bs / 1e6:.0f} MB{Fore.RESET}'
    elif Bs > 1e3:
        return f'{Fore.LIGHTYELLOW_EX}{Bs / 1e3:.0f} KB{Fore.RESET}'
    else:
        return f'{Fore.LIGHTBLUE_EX_EX}{Bs} B{Fore.RESET}'

def pickle_wrap(filename, callback, easy_override=False):
    if os.path.isfile(filename) and not easy_override:
        print(f'pickle_wrap loading: {Fore.LIGHTYELLOW_EX}{ntpath.split(filename)[1]}{Fore.RESET}, '
              f'File size {get_Bs_str(os.path.getsize(filename))}')
        with open(filename, "rb") as file:
            print('Loaded')
            return pickle.load(file)
    else:
        print(f'pickle wrap doing: {callback} | {filename=}')
        output = callback()
        print('Callback done')
        with open(filename, "wb") as new_file:
            pickle.dump(output, new_file)
        print(f'pickle_wrap dumped: {Fore.LIGHTYELLOW_EX}{ntpath.split(filename)[1]}{Fore.RESET}, '
              f'File size {get_Bs_str(os.path.getsize(filename))}')
        return output

def abline(slope, intercept, subplot_ax):

    x_vals = np.array(subplot_ax.get_xlim())
    y_vals = intercept + slope * x_vals
    subplot_ax.plot(x_vals, y_vals, '-')

def flatten(itr, l=False):
    # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    if l:
        t = list()
    else:
        t = tuple()
    for e in itr:
        if isinstance(e, str):
            t += (e, )
            continue
        try:
            t += flatten(e)
        except:
            t += (e,)
    return t

def flatten_lineups(list_itr, l=False, as_itr=False):
    return_list = []
    for itr in list_itr:
        if as_itr:
            yield flatten(itr, l=l)
        else:
            return_list.append(flatten(itr, l=l))
    if not as_itr: return return_list

def rando_drop(l, p_drop):
    l_pruned = []
    for item in l:
        if random() > p_drop:
            l_pruned.append(item)
    print('\tRemaining after random drop of {p:.0%}: {n:.1e}'.format(p=p_drop, n=len(l_pruned)))
    return l_pruned

def partition (list_in, n):
    shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def forcibly_sort(ar_out, max_rank):
    ar_out_high_on_top = np.sort(ar_out, axis=0)[::-1, :]
    ar_out_high_on_bottom = np.sort(ar_out, axis=0)
    where_flip = ar_out[0, :] > ar_out[max_rank-1, :]
    ar_out_ordered = np.append(ar_out_high_on_bottom[:, ~where_flip], ar_out_high_on_top[:, where_flip], axis=1)
    #ar_out = np.array([ar_out, ar_out_ordered]).mean(axis=0)
    return ar_out_ordered


def gmean(ar, axis=0):
    ar_log = np.log(ar)
    return np.exp(ar_log.mean(axis=axis))

def empty_gen():
    yield from ()