import os

import dill
import numpy as np

from make_distributions import plot as plot


class Expert_Distribution:
    '''
    Object that holds the expert distribution for a position.
    Not very vital for understanding how the project works. It mostly just holds
        variables and don't execute many operations.
    '''
    def __init__(self, experts, ar_all_density, x_density_pts, x_cum_tiles, pos,
                 max_rank, year_st=None, year_end=None,
                 fn=None, plot_name=''):
        self.experts = experts
        self.ar_all_density = ar_all_density # (experts, rank, pts) -> density
        self.x_density_pts = x_density_pts
        self.x_cum_tiles = x_cum_tiles
        self.pos = pos
        self.ar_mean_density = self.ar_all_density.mean(axis=0)
        self.max_rank = max_rank
        self.year_st = year_st # Data are included from [year st, year end]
        self.year_end = year_end # If year_end == 2020, then 2020 data included
        self.fn = fn
        self.params_full = None
        self.out_dir = r'make_distributions'

        self.test_yr = None
        self.score = None
        self.cnt = None
        self.test_yrs = None
        self.scores = None
        self.cnts = None

        self.effect_of_year = None
        self.effect_if_worse_abs = None
        self.logged = False
        self.plot_name = plot_name

    def merge_w_ED(self, ED1):
        if self.max_rank > ED1.max_rank:
            self.ar_mean_density[:ED1.max_rank] += ED1.ar_mean_density
            self.ar_mean_density[:ED1.max_rank] /= 2

    def prune_ar(self, max_rank, dont_touch_ar=False):
        self.max_rank = max_rank
        if dont_touch_ar: return
        self.ar_mean_density = self.ar_mean_density[:max_rank]
        self.ar_all_density = self.ar_all_density[:, :max_rank]

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def set_fn(self, fn):
        self.fn = fn

    def save(self):
        fp = os.path.join(self.out_dir, 'EDs', f'{self.fn}.pkl')
        print(f'Save ED: {fp=}')
        d = dill.dumps(self)
        with open(fp, 'wb') as file:  # dill converts objects into bytestrings
            dill.dump(d, file)
        return True

    def plot_density(self, save_plot=False, cumulative_density=False,
                     save_fp=None, max_rank=None):
        if save_fp is not None:
            fp = save_fp
        elif cumulative_density:
            fp = os.path.join(self.out_dir, 'figs', f'{self.fn}_cumsum.png')
        else:
            fp = os.path.join(self.out_dir, 'figs', f'{self.fn}.png')
        max_rank = self.max_rank if max_rank is None else max_rank
        ar_mean_density = self.ar_mean_density[:max_rank]
        plot.plot_density(ar_mean_density, self.x_density_pts,
                          fp=fp, save_plot=save_plot, pos=self.pos,
                          cumdens=cumulative_density,
                          extra_str=self.plot_name)

    def convert_to_log_density(self):
        if self.logged:
            print('Already logged!')
            return
        self.ar_all_density = np.log(self.ar_all_density)
        self.logged = True

    def unlog(self):
        if not self.logged:
            print('Already not-logged!')
            return
        self.ar_all_density = np.exp(self.ar_all_density)
        self.logged = False

    def set_score(self, test_yr, score, cnt):
        self.test_yr = test_yr
        self.score = score
        self.cnt = cnt

    def set_scores_by_year(self, years, scores, cnts):
        self.test_yrs = years
        self.scores = scores
        self.cnts = cnts

    def set_effect_of_year(self, effect_of_year):
        self.effect_of_year = effect_of_year

    def set_effect_if_worse_abs(self, effect_if_worse_abs):
        self.effect_if_worse_abs = effect_if_worse_abs
