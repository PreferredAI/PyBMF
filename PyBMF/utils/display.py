import matplotlib.pyplot as plt
import ctypes
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from matplotlib import cm, colormaps, MatplotlibDeprecationWarning
from .sparse_utils import to_triplet, to_dense
import platform
import warnings


def show_matrix(settings, 
                scaling=1.0, ppi=96, hds=1.5, pixels=None, 
                title=None, fontsize=8, 
                keep_nan=True, 
                colorbar=False, clim=None, discrete=False, center=True, 
                cmap='rainbow', cmin='gray', cmax='black', cnan='white', 
                save_fig=True):
    '''Show the matrix and factors.

    Parameters
    ----------
    settings : list of tuple
        A list of (data, location, title) tuple.
    scaling : float, default: 1.0
        The scaling factor. The default `scaling` is 1.0, the maximum size a figure can be displayed within the screen.
    ppi : int, default: 96
        Pixels per inch. The `ppi` of a 4K 24" display is 96.
    hds : float, default: 1.5
        High DPI scaling, if your Python IDE supports this. The default `hds` in Spyder is 1.5.
    pixels : int, optional
        Each cell in a matrix takes up `pivels` * `pixels` on screen. This will overwrite `scaling`.
    title : string, optional
        The centered suptitle of the figure.
    fontsize : int, default: 8
        Size of the `title` and subtitles.
    colorbar : bool, default: False
        Show colorbar.
    clim : list, optional
        Colorbar range limit applied over all matrices. If `clim` is `None`, each matrix will have its own colorbar range limit separately.
    discrete : bool, default: False
        Show discrete colorbar.
    center : bool, default: True
        Available only when `discrete` is True.
    cmap : str, default: 'rainbow'
        The colormap.
    cmin : str, default: 'gray'
        The color of values lower than the range limit `clim`.
    cmax : str, default: 'black'
        The color of values higher than the range limit `clim`.
    cnan : str, default: 'white'
        The color of `NaN`. To differentiate real zeros and `NaN` in sparse marices.
    save_fig : bool, default: True
        Save the figure to output directory.
    '''
    rows = []       # row index of each matrix
    cols = []       # col index of each matrix
    widths = {}     # width of matrices at each col in grid
    heights = {}    # height of matrices at each row in grid

    for data, location, description in settings:
        r, c = location
        rows.append(r)
        cols.append(c)
        
        if r not in heights.keys():
            heights[r] = data.shape[0]
        if c not in widths.keys():
            widths[c] = data.shape[1]

    n_rows = max(rows) + 1 # num of rows of the grid
    n_cols = max(cols) + 1 # num of cols of the grid

    for r in range(n_rows):
        if r not in heights.keys():
            heights[r] = 0 # in case there's no matrix on this row
    for c in range(n_cols):
        if c not in widths.keys():
            widths[c] = 0 # in case there's no matrix on this column

    if colorbar is False:
        '''Without colorbar
        '''
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw={
            'width_ratios': [widths[c] for c in range(n_cols)],
            'height_ratios': [heights[r] for r in range(n_rows)]
        })

        if n_rows == 1 or n_cols == 1:
            axes = np.reshape(axes, [n_rows, n_cols]) # axes must be a 2d array

        for data, location, description in settings:
            r, c = location
            # if keep_nan:
                # data = fill_nan(data, mask=data)
            data = to_dense(data, keep_nan=keep_nan)
            im = axes[r, c].matshow(data, cmap=cmap)
            axes[r, c].set_title(description, fontdict={'fontsize': fontsize}, loc='left')
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

        # set the rest of subplots to invisible
        mat_locs = [(r, c) for r, c in zip(rows, cols)]
        all_locs = [(r, c) for r in range(n_rows) for c in range(n_cols)]  
        nan_locs = set(all_locs) - set(mat_locs)

    else:
        '''With colorbar and advanced color settings
        '''
        cbar_width = 5
        heights_with_cbar = [cbar_width] * (len(heights) * 2)
        widths_with_cbar = [cbar_width] * (len(widths) * 2)

        for r in range(len(heights)):
            heights_with_cbar[r * 2] = heights[r]
        for c in range(len(widths)):
            widths_with_cbar[c * 2] = widths[c]

        n_rows *= 2
        n_cols *= 2

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw={
            'width_ratios': [widths_with_cbar[c] for c in range(n_cols)],
            'height_ratios': [heights_with_cbar[r] for r in range(n_rows)]
        })

        if n_rows == 1 or n_cols == 1:
            axes = np.reshape(axes, [n_rows, n_cols]) # axes must be a 2d array

        mat_locs = []
        cbar_locs = []

        for data, location, description in settings:
            r, c = location
            r *= 2
            c *= 2
            # if keep_nan:
            #     data = fill_nan(data, mask=data)
            data = to_dense(data, keep_nan=keep_nan)
            dmin, dmax = (np.nanmin(data), np.nanmax(data)) if clim is None else (clim[0], clim[1])

            if discrete:
                cnum = dmax - dmin + (1 if center else 0)
            else:
                center = False
                cnum = None

            vmin = dmin - (0.5 if center else 0)
            vmax = dmax + (0.5 if center else 0)
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            with warnings.catch_warnings():
                try:
                    cmap = cm.get_cmap(cmap, cnum).copy()
                except Warning as w:
                    cmap = plt.get_cmap(cmap, cnum)
                    # to be deprecated?
                except Exception as e:
                    cmap = plt.get_cmap(cmap, cnum)
            cmap.set_under(cmin)
            cmap.set_over(cmax)
            cmap.set_bad(cnan)

            im = axes[r, c].matshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
            axes[r, c].set_title(description, fontdict={'fontsize': fontsize}, loc='left')
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

            # todo: fix height mismatch
            mat_locs.append((r, c))
            ticks = np.arange(dmin, dmax + 1) if isinstance(dmin, int) and isinstance(dmax, int) else None
            emin, emax = np.nanmin(data) < dmin, np.nanmax(data) > dmax
            if emin and emax:
                extend = 'both'
            elif emin and not emax:
                extend = 'min'
            elif not emin and emax:
                extend = 'max'
            else:
                extend = 'neither'
            
            if data.shape[0] >= data.shape[1]:
                plt.colorbar(im, cax=axes[r, c + 1], ticks=ticks, extend=extend, orientation='vertical')
                cbar_locs.append((r, c + 1))
            else:
                plt.colorbar(im, cax=axes[r + 1, c], ticks=ticks, extend=extend, orientation="horizontal")
                cbar_locs.append((r + 1, c))

        # set the rest of subplots to invisible
        all_locs = [(r, c) for r in range(n_rows) for c in range(n_cols)]  
        nan_locs = set(all_locs) - set(mat_locs) - set(cbar_locs)

    for r, c in nan_locs:
        axes[r, c].set_visible(False)
        
    # display
    width_inches, height_inches = get_size_inches(
        scaling=scaling,
        ppi=ppi,
        hds=hds, 
        pixels=pixels,
        width_cells=sum(widths.values()),
        height_cells=sum(heights.values()))

    fig.set_size_inches(width_inches, height_inches)
    
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    fig = plt.gcf() 
    plt.show(block=False)
    plt.draw()

    if save_fig:
        fig.savefig("../experiment/display.pdf", dpi=100)


def get_size_inches(scaling, ppi, hds, pixels, width_cells, height_cells):
    '''Get figure size in inches.

    Parameters
    ----------
    width_cells : int
        Figure width in the number of matrix cells.
    height_cells : int
        Figure height in the number of matrix cells.

    Returns
    -------
    width_inches : float
        Figure width in inches.
    height_inches : float
        Figure height in inches.
    '''
    if pixels is None:
        # get screen resolution in pixels
        if platform.system() == "Windows":
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            screen_width_pixels = user32.GetSystemMetrics(0)
            screen_height_pixels = user32.GetSystemMetrics(1)
        else: # todo: linux
            screen_width_pixels = 1920
            screen_height_pixels = 1080
        
        # check which dimension should be aligned
        screen_aspect_ratio = screen_width_pixels / screen_height_pixels
        figure_aspect_ratio = width_cells / height_cells
        if screen_aspect_ratio > figure_aspect_ratio:
            # match the height of screen
            height_inches = screen_height_pixels * scaling / ppi / hds
            width_inches = height_inches * figure_aspect_ratio
        else:
            # match the width of screen
            width_inches = screen_width_pixels * scaling / ppi / hds
            height_inches = width_inches / figure_aspect_ratio
    else:
        # set the size according to presumed pixels
        width_inches = width_cells * pixels * scaling / ppi / hds
        height_inches = height_cells * pixels * scaling / ppi / hds
        
    return (width_inches, height_inches)


def fill_nan(X, mask: spmatrix):
    '''Fill the missing values of a sparse matrix with NaN, so that missing values in a sparse matrix are displayed differently from zeros.

    Used for displaying matrices while identifying missing values.

    Parameters
    ----------
    X : ndarray or spmatrix
        The matrix with values to be filled with `NaN`.
    mask : spamtrix
        The masking matrix.
        Explicit zeros in `mask` are not considered as missing. 
        Note that there are several ways to preserve zeros in a sparse matrix. BaseSplit.load_neg_data() is one fot them.

    Returns
    -------
    Y : ndarray
        The dense matrix with `NaN` in it.
    '''
    rows, cols, _ = to_triplet(mask)
    Y = np.empty(shape=X.shape)
    Y.fill(np.nan)
    for i in range(len(rows)):
        Y[rows[i], cols[i]] = X[rows[i], cols[i]]
    return Y


def show_factor_distribution(U, V, resolution=100, show_hist=False, show_minmax=True, remove_below=None, us=None, vs=None):
    '''Show the distribution of real-valued factor matrices U and V.

    
    '''
    k = U.shape[1]
    
    my_dpi = 96

    if show_hist:
        fig, axs = plt.subplots(k, 2, sharex=True, sharey=False, tight_layout=True, figsize=(800/my_dpi, 40*k/my_dpi), dpi=my_dpi)
        
    for i in range(k):
        if show_hist:
            u = to_dense(U[:, i])
            if remove_below is not None:
                u = u[u > remove_below]
            axs[i, 0].hist(u, bins=resolution)
            axs[i, 0].set_xticks([])
            axs[i, 0].set_yticks([])
            if us is not None:
                axs[i, 0].axvline(x=us[i], color='r', linestyle='-')

            v = to_dense(V[:, i])
            if remove_below is not None:
                v = v[v > remove_below]
            axs[i, 1].hist(v, bins=resolution)
            axs[i, 1].set_xticks([])
            axs[i, 1].set_yticks([])
            if vs is not None:
                axs[i, 1].axvline(x=vs[i], color='r', linestyle='-')

        if show_minmax:
            print('{} - [{:.3f}, {:.3f}], [{:.3f}, {:.3f}]'.format(i, U[:, i].min(), U[:, i].max(), V[:, i].min(), V[:, i].max()))

    if show_hist:
        plt.show()