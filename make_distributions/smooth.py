import numpy as np
import scipy
#from matplotlib import pyplot as plt
from tqdm import tqdm

#import make_distributions.plotting as plot
from make_distributions.plot import plot_density

import skfda
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch
from skfda.preprocessing.smoothing import kernel_smoothers


def smooth_horizontal(ar, size=15, mode='nearest',
                      cv_params=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50)):
    '''
    Smooths the horizontal axis of array ar. If mode='CV', then it uses
        cross-validation to find the optimal smoothing bandwidth.
    '''
    for rank0 in range(ar.shape[0]):
        if mode == 'CV':
            cv_params = list(cv_params)
            x = range(len(ar[rank0, :]))
            fd = skfda.FDataGrid(ar[rank0, :], x)
            grid = SmoothingParameterSearch(kernel_smoothers.KNeighborsSmoother(
                kernel=skfda.misc.kernels.normal), cv_params)
            grid.fit(fd)
            print(f'Smooth horizontal optimal bandwidth: {grid.best_params_}')
            fd_new = grid.transform(fd)
            ar[rank0, :] = fd_new.data_matrix[0, :, 0]
        else:
            ar[rank0, :] = scipy.ndimage.gaussian_filter1d(ar[rank0, :], size,
                                                           mode=mode)
    return ar


def smooth_vertical(ar, size=15, mode='nearest', correct=False, do_plot=False):
    '''
    Smooths the vertical axis of an array. Applies the correction factor
        to the smoothed array if correct=True. The correction factor is
        described in the white paper.
    '''
    pad_size = int(25)

    ar_padded = []
    ar_padded_smooth = []
    for tile in tqdm(range(ar.shape[1]), desc='Smoothing tile-by-tile'):
        if correct:
            slope, intercept, r_value, p_value, std_err = \
                scipy.stats.linregress(range(ar.shape[0])[:10], ar[:10, tile])
            ar_tile_padded = np.append(np.empty(pad_size), ar[:, tile])
            ar_tile_padded[:pad_size] = intercept + slope * range(-pad_size, 0)

            # ar_tile_padded[:pad_size] = horz_smooth(ar_tile_padded[np.newaxis, :pad_size], 30)[0]
            ar_tile_padded[ar_tile_padded < 1e-20] = 1e-20
            ar_padded.append(ar_tile_padded)
            # print(f'{len(ar_tile_padded)=}')

            ar_tile_padded_smooth = scipy.ndimage.gaussian_filter(ar_tile_padded, size, mode=mode)
            ar_padded_smooth.append(ar_tile_padded_smooth)
            ar[:, tile] = ar_tile_padded_smooth[pad_size:]
        else:
            ar[:, tile] = scipy.ndimage.gaussian_filter(ar[:, tile], size, mode=mode)

    if do_plot:
        ar_padded = np.array(ar_padded).transpose()
        fp = r'E:\Weekly_Distributions\figs\RB\max_48\ar_padded.png'
        plot_density(ar_padded, np.linspace(0, 50, ar.shape[1]),
                     num_super_ranks=pad_size, fp=fp, save_plot=True,
                     extra_str=' (Padded upper ranks)')

        ar_padded_smooth = np.array(ar_padded_smooth).transpose()
        fp = r'E:\Weekly_Distributions\figs\RB\max_48\ar_padded_smooth.png'
        plot_density(ar_padded_smooth, np.linspace(0, 50, ar.shape[1]),
                     num_super_ranks=pad_size, fp=fp, save_plot=True,
                     extra_str=' (Padded upper ranks smooth)')
        quit()

    return ar


def normalize_horizontal(ar):
    '''
    Normalizes the horizontal axis of an array so that each row sums to 1.
    '''
    assert np.min(ar) > -.00000001, 'Can\'t normalize log-scaled ar'
    return ar / ar.sum(axis=1)[:, None]
