import matplotlib.pyplot as plt
import ctypes
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from matplotlib import cm
from .sparse_utils import to_triplet, to_dense


def show_matrix(settings, 
                scaling=1.0, ppi=96, hds=1.5, pixels=None, 
                title=None, fontsize=8, 
                keep_nan=True, 
                colorbar=False, clim=None, discrete=False, center=True, 
                cmap='rainbow', cmin='gray', cmax='black', cnan='white'):
    """Show the matrix and factors

    Parameters
    ----------
    settings: list of tuples
        a list of (data, location, title) tuple.
    scaling: float, optional
        scaling factor. the default is 1.0, which opens a window that match the height or width of the screen.
    ppi: int, optional
        pixels per inch. the default, tested on a 4K 24" screen, is 96.
    hds : float, optional
        high DPI scaling if your IDE support this. e.g. the default in Spyder is 1.5.
    pixels : int, optional
        each cell in a matrix takes up pivels * pixels on screen. this will overwrite scaling.
    title : string, optional
        name of each matrix.
    fontsize : int
        size of titles.
    colorbar : bool
        whether to enable colorbar and the advanced settings including limitation, discretization and centering.
    cmap : str
        name of colormap.
    clim : list
        shared range limit of colorbar, applied to all matrices.
        will show colorbar separately if clim is None.
    discrete : bool
        show discrete colorbar.
    center : bool
        available only when discrete is True.
    """
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
<<<<<<< HEAD
                plt.colorbar(im, cax=axes[r + 1, c], ticks=ticks, extend=extend, orientation="horizontal")
=======
                # plt.colorbar(im, cax=axes[r + 1, c], location='bottom')
                plt.colorbar(im, cax=axes[r + 1, c], orientation="horizontal")
>>>>>>> 8ea583386c050f827fd03c38c626ea0e080fd29f
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
        figure_width_cells=sum(widths.values()),
        figure_height_cells=sum(heights.values())
        )

    fig.set_size_inches(width_inches, height_inches)
    
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)
        
    plt.show(block=False)


def get_size_inches(scaling, ppi, hds, pixels, figure_width_cells, figure_height_cells):
    """Get fig size in inches

    Parameters
    ----------
    scaling : TYPE
        DESCRIPTION.
    ppi : TYPE
        DESCRIPTION.
    hds : TYPE
        DESCRIPTION.
    pixels : TYPE
        DESCRIPTION.
    figure_width_cells : TYPE
        Approximate number of all cells horizontally.
    figure_height_cells : TYPE
        Approximate number of all cells vertically.

    Returns
    -------
    width_inches : TYPE
        DESCRIPTION.
    height_inches : TYPE
        DESCRIPTION.

    """
    if pixels is None:
        # get screen resolution in plxels
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        screen_width_pixels = user32.GetSystemMetrics(0)
        screen_height_pixels = user32.GetSystemMetrics(1)
        
        # check which dimension should be aligned
        screen_aspect_ratio = screen_width_pixels / screen_height_pixels
        figure_aspect_ratio = figure_width_cells / figure_height_cells
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
        width_inches = figure_width_cells * pixels * scaling / ppi / hds
        height_inches = figure_height_cells * pixels * scaling / ppi / hds
        
    return (width_inches, height_inches)


def fill_nan(X, mask: spmatrix):
    '''Fill the missing values of a sparse matrix with NaN

    So that missing values are displayed differently from zeros.

    Explicit zeros in the mask are not considered as missing.
    Explicit zeros can be added properly into a sparse matrix as in BaseSplit.load_neg_data().
    Used for displaying matrices to separate missing values.
    '''
    rows, cols, _ = to_triplet(mask)
    Y = np.empty(shape=X.shape)
    Y.fill(np.nan)
    for i in range(len(rows)):
        Y[rows[i], cols[i]] = X[rows[i], cols[i]]
    return Y