import numpy as np
import scipy
from tqdm import tqdm
import skfda

from make_distributions.plot import plot_density
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch
from skfda.preprocessing.smoothing import kernel_smoothers


def smooth_horizontal(ar_density, size=15, mode='nearest',
                      cv_params=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50)):
    '''
    Smooths the horizontal axis of array ar. If mode='CV', then it uses
        cross-validation to find the optimal smoothing bandwidth.
        skfda package used to manage finding the optimal bandwidth.

    :ar_density: 2D matrix representing the distributions. See density.py
    :size: Bandwidth of the smoothing kernel. Only used if mode != 'CV'
    :mode: Parameter for scipy.ndimage.gaussian_filter(...). Determines how the
        input array is extended when the filter overlaps a border.
        If mode == 'CV', will use cross-validation to find the optimal bandwidth
    :cv_params: Bandwidths tested in cross-validation. Only used if mode == 'CV'
    :ar_density: Smoothed density array
    '''
    for rank0 in range(ar_density.shape[0]):
        if mode == 'CV':
            cv_params = list(cv_params)
            x = range(len(ar_density[rank0, :]))
            fd = skfda.FDataGrid(ar_density[rank0, :], x)
            grid = SmoothingParameterSearch(kernel_smoothers.KNeighborsSmoother(
                kernel=skfda.misc.kernels.normal), cv_params)
            grid.fit(fd)
            print(f'Smooth horizontal optimal bandwidth: {grid.best_params_}')
            fd_new = grid.transform(fd)
            ar_density[rank0, :] = fd_new.data_matrix[0, :, 0]
        else:
            ar_density[rank0, :] = scipy.ndimage.gaussian_filter1d(
                ar_density[rank0, :], size, mode=mode)
    return ar_density


def smooth_vertical(ar_density, size=15, mode='nearest', correct=False,
                    do_plot=False):
    '''
    Smooths the vertical axis of an array.
    If correct=True, this code applies the correction described in section
        1.2.4 of the white paper. The correction uses a linear regression to
        simulate hypothetical players ranked better than the actual best player.
        Including these hypothetical players in the smoothing process helps
        avoid biases related to the edges.
    If do_plot=True, this function will plot the distribution, which is useful
        if correct=True, and you want to visualize the hypothetical
        (see Figure 5 in the white paper).

    :ar_density: 2D matrix representing the distributions. See density.py
    :size: Bandwidth of the smoothing kernel
    :mode: Parameter for scipy.ndimage.gaussian_filter(...). Determines how the
        input array is extended when the filter overlaps a border.
    :correct: If True, applies the correction above.
    :do_plot: If True, plots the distribution.
    :ar_density: Smoothed density array
    '''
    pad_size = int(25)

    ar_padded = []
    ar_padded_smooth = []
    for tile in tqdm(range(ar_density.shape[1]), desc='Smoothing tile-by-tile'):
        if correct:
            slope, intercept, r_value, p_value, std_err = \
                scipy.stats.linregress(range(ar_density.shape[0])[:10],
                                       ar_density[:10, tile])
            ar_tile_padded = np.append(np.empty(pad_size), ar_density[:, tile])
            ar_tile_padded[:pad_size] = intercept + slope * range(-pad_size, 0)

            ar_tile_padded[ar_tile_padded < 1e-20] = 1e-20
            ar_padded.append(ar_tile_padded)

            ar_tile_padded_smooth = scipy.ndimage.gaussian_filter(
                ar_tile_padded, size, mode=mode)
            ar_padded_smooth.append(ar_tile_padded_smooth)
            ar_density[:, tile] = ar_tile_padded_smooth[pad_size:]
        else:
            ar_density[:, tile] = scipy.ndimage.gaussian_filter(
                ar_density[:, tile], size, mode=mode)

    if do_plot:
        ar_padded = np.array(ar_padded).transpose()
        fp = r'E:\Weekly_Distributions\figs\RB\max_48\ar_padded.png'
        plot_density(ar_padded, np.linspace(0, 50, ar_density.shape[1]),
                     num_super_ranks=pad_size, fp=fp, save_plot=True,
                     extra_str=' (Padded upper ranks)')

        ar_padded_smooth = np.array(ar_padded_smooth).transpose()
        fp = r'E:\Weekly_Distributions\figs\RB\max_48\ar_padded_smooth.png'
        plot_density(ar_padded_smooth, np.linspace(0, 50, ar_density.shape[1]),
                     num_super_ranks=pad_size, fp=fp, save_plot=True,
                     extra_str=' (Padded upper ranks smooth)')

    return ar_density