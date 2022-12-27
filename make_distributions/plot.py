import os
from pathlib import Path

import numpy as np
from scipy import  ndimage
import matplotlib
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors


def plot_density(ar_density, x_pts, pos='RB', extra_str='', save_plot=False,
                 fp=None, cumdens=False, black10=False, num_super_ranks=0):
    '''
    Plots the density and cumulative density illustrations (Figures 2-7)
        throughout the white paper

    :param ar_density: 2D density array, where axis=0 is the rank,
                       and axis=1 indexes the points scored
    :param x_pts: Points scored covered by axis 1 of the density
                  so if x_pts = np.linspace(0, 50, 1000), then
                  ar_density.shape[1] = 1000 and column j corresponds to
                  x_pts[j] points scored
    :param pos: Position (str)
    :param extra_str: string added to the filename of the saved plot
    :param save_plot: If False, it opens the plot in a window. If True, saves
    :param fp: Output filepath, if None, creates a filepath
    :param cumdens: (Cumulative density)
        If False, plots density. If True, plots cumulative density
    :param black10: If True, plots ranks % 10 == 0 in black.
        black10 was not used for the white paper figures
    :param num_super_ranks: For plotting below zero hypothetical ranks
                            (see white paper Figure 5)
    '''
    fig_dir = r'make_distributions\figs'

    print(f'{x_pts=}')
    if num_super_ranks > 0:
        space = np.linspace(0, 1, 3000)
        colors_ar = plt.cm.turbo_r(space)
        vmin = -24
        vmax = 60
        pos1 = int((-vmin + 0.4 + 1) / (vmax - vmin) * space.shape[0])
        posneg1 = int((-vmin - 0.65 + 1) / (vmax - vmin) * space.shape[0])
        colors_ar[posneg1:pos1] = [0, 0, 0, 1]
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                                      colors_ar)
    else:
        cmap = matplotlib.cm.get_cmap('turbo_r')

    print(f'{ar_density.shape=}')
    if cumdens:
        for rank0 in range(ar_density.shape[0] - 1, -1, -1):

            color = cmap(rank0 / ar_density.shape[0])
            ar_cumsum = np.cumsum(ar_density[rank0, :] * 100)
            plt.plot(ar_cumsum, x_pts, c=color, linewidth=0.95)
        if black10:
            for rank0 in range(0, ar_density.shape[0], 10):
                plt.plot(np.cumsum(ar_density[rank0, :] * 100), x_pts, c='k',
                         linewidth=0.95)
        plt.xlabel('Percentile', fontsize=16)
        plt.ylabel('Fantasy Points', fontsize=16)
        ax = plt.gca()
        ax.set_xlim((0, 100))
        ax.set_ylim((np.min(x_pts), np.max(x_pts)))
        if np.min(x_pts) < 0:
            plt.plot([0, 100], [0, 0], c='k', linewidth=0.95)

    else:
        for rank0 in range(ar_density.shape[0] - 1, -1, -1):
            color = cmap(rank0 / ar_density.shape[0])
            plt.plot(x_pts, ar_density[rank0, :] * 100, c=color,
                     linewidth=0.95)
        if black10:
            for rank0 in range(0, ar_density.shape[0], 10):
                plt.plot(x_pts, ar_density[rank0, :] * 100, c='k',
                         linewidth=0.95)
        plt.ylabel('Density', fontsize=16)
        plt.xlabel('Fantasy Points', fontsize=16)
        ax = plt.gca()
        ax.set_xlim((np.min(x_pts), np.max(x_pts)))
        plt.yticks([.0, .05, .1, .15, .2, .25])
        ax.set_ylim((0, 0.29))
        if np.min(x_pts) < 0:
            plt.plot([0, 0], [np.min(x_pts), np.max(x_pts)], c='k',
                     linewidth=0.95)


    if num_super_ranks > 0: # For plotting sub-zero hypothetical ranks (Fig. 5)
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,
                                              norm=Normalize(vmin=-24, vmax=60),
                                              ), ticks=[-24, -12, 1, 12,
                                                        24, 36, 48, 60])
        cbar.ax.set_yticklabels([f'{pos}(-24)', f'{pos}(-12)', f'{pos}1',
                                 f'{pos}12', f'{pos}24', f'{pos}36', f'{pos}48',
                                 f'{pos}60'])

    elif ar_density.shape[0] == 48: # For plotting RBs/WRs up to 48
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,
                                              norm=Normalize(vmin=1,
                                                             vmax=ar_density.shape[0]),
                                              ), ticks=[1, 12, 24, 36, 48])
        cbar.ax.set_yticklabels([f'{pos}1', f'{pos}12', f'{pos}24', f'{pos}36',
                                 f'{pos}48'])
    elif ar_density.shape[0] == 24: # For plotting QBs/TEs/DSTs up to 24
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,
                                              norm=Normalize(vmin=1,
                                                             vmax=ar_density.shape[0]),
                                              ), ticks=[1, 6, 12, 18, 24])
        cbar.ax.set_yticklabels([f'{pos}1', f'{pos}6', f'{pos}12', f'{pos}18',
                                 f'{pos}24'])
    else:
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap,
                                              norm=Normalize(
                                                  vmin=1,
                                                  vmax=ar_density.shape[0]),
                                              ))
    cbar.ax.invert_yaxis()
    title = f'Position: {pos}{extra_str}'
    plt.title(title, fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tick_params(colors='lightgray', size=0.5)
    plt.xticks(fontsize=13, color='k')
    plt.yticks(fontsize=13, color='k')
    plt.grid(zorder=1, linewidth=0.5)

    if save_plot:
        print('SAVING PLOT:')

        if fp is None: fp = os.path.join(fig_dir,
                            title.replace(':', '').replace(' ', '_') + '.png')
        print(f'{os.path.join(os.getcwd(), fp)=}')
        print(f'{fp=}')
        plt.tight_layout()

        plt.savefig(fp, dpi=400)
        plt.show()
        plt.clf()
    else:
        plt.show()


def plot_pre_density(ar, pos, max_rank):
    '''
    Used for plotting the raw data before estimating density
        Creates scatter plot where x = points scores, and y = player rank

    :param ar: matrix [i, t] of points scored by the ith ranked player at time t
    :param pos: Position (str)
    :param max_rank: worst rank to plot (higher rank = worse)
    '''
    cmap = matplotlib.cm.get_cmap('turbo_r')
    print(f'{max_rank=}')
    mean_line = []
    lower_line = []
    upper_line = []
    plt.figure(figsize=(7.4, 5.8))
    for rank0 in range(max_rank-12):
        plt.scatter(ar[rank0, :], [rank0]*ar.shape[1], color='k',
                    alpha=0.8, s=0.8, zorder=1000)
        l = ar[rank0, :]
        m = np.nanmedian(ar[rank0, :])
        l = l[~np.isnan(l)]
        l = np.sort(l)
        lower_line.append(l[int(len(l)*0.1)])
        upper_line.append(l[int(len(l)*0.9)])
        mean_line.append(m)

    lower_line = ndimage.gaussian_filter1d(lower_line, 2)
    upper_line = ndimage.gaussian_filter1d(upper_line, 2)
    mean_line = ndimage.gaussian_filter1d(mean_line, 2)

    for rank0, (lower, upper) in enumerate(zip(lower_line, upper_line)):
        color = cmap(rank0 / max_rank)
        plt.plot([0, lower], [rank0, rank0], linewidth=1, color=color, zorder=2,
                 alpha=1-abs(rank0 / max_rank - 0.5))
        plt.plot([upper, 50], [rank0, rank0], linewidth=1, color=color, zorder=2,
                 alpha=1-abs(rank0 / max_rank - 0.5))

    plt.gca().fill_betweenx(range(len(mean_line)),
                            lower_line, upper_line, zorder=100, alpha=0.12,
                            color='k')
    plt.plot(mean_line, range(len(mean_line)),
             color='k', linewidth=2.5, zorder=3)
    print(mean_line)
    plt.xlim(0, 50)
    plt.ylim(max_rank-12-1+.15, -.15)
    plt.title(f'Position: {pos}', fontsize=16)
    plt.xlabel('Fantasy Points', fontsize=16)
    plt.ylabel('Rank', fontsize=16)
    plt.yticks([0, 11, 23, 35, 47],
               [f'{pos}1', f'{pos}12', f'{pos}24', f'{pos}36', f'{pos}48'],
               fontsize=13)
    plt.xticks(fontsize=13)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(fr'{pos}_density.png', dpi=300)
    plt.show()


def save_ED_plot(ED, test_size, params_fn, test_acc, max_rank,
                 single_year_test, test_st, pos='rb'):
    '''
    Calls plot_density(...) for the Expert Distribution (ED) passed
        Most of the parameters are just used to specify the location where the
        plot is saved and what the filename will be

    :param ED: Expert Distribution
    :param test_size: size of test set
    :param params_fn: filename of the parameters used to generate the ED
    :param test_acc: testing accuracy
    :param max_rank: highest rank to plot
    :param single_year_test: If True, then the test set is from the specified
        year and thus that year wasn't used to fit the ED. The resulting plot
        will be saved in a directory corresponding to that test year
    :param test_st: If single_year_test is True, then this is the year tested
    '''
    fn = f'acc{test_acc:.3f}_cnt{test_size}_{params_fn}'
    fn_dir = f'{pos}\\max_{max_rank}'
    if single_year_test:
        fn_dir = os.path.join(str(test_st), fn_dir)
    out_dir = r'/make_distributions'
    Path(os.path.join(out_dir, 'figs', fn_dir)).mkdir(parents=True,
                                                      exist_ok=True)
    Path(os.path.join(out_dir, 'EDs', fn_dir)).mkdir(parents=True,
                                                     exist_ok=True)
    fn = os.path.join(fn_dir, fn)
    ED.set_fn(fn)
    ED.plot_density(save_plot=True, max_rank=max_rank)
    ED.plot_density(save_plot=True, cumdens=True, max_rank=max_rank)
    ED.save()