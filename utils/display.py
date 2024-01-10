import matplotlib.pyplot as plt
import ctypes
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def show_matrix(settings, scaling=1.0, ppi=96, hds=1.5, pixels=None, title=None, fontsize=8, colorbar=False):
    """Show the matrix and factors

    Parameters
    ----------
    settings: tuple
        data and location and title.
    scaling: float, optional
        scaling factor. the default is 1.0, which opens a window that match the height or width of the screen.
    ppi: int, optional
        pixels per inch. the default, tested on a 4K 24" screen, is 96.
    hds : float, optional
        high DPI scaling if your IDE support this. e.g. the default in Spyder is 1.5.
    pixels : int, optional
        each cell in a matrix takes up pivels * pixels on screen. this will overwrite scaling.
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
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw={
            'width_ratios': [widths[c] for c in range(n_cols)],
            'height_ratios': [heights[r] for r in range(n_rows)]
        })

        if n_rows == 1 or n_cols == 1:
            axes = np.reshape(axes, [n_rows, n_cols]) # axes must be a 2d array

        for data, location, description in settings:
            r, c = location
            im = axes[r, c].matshow(data)
            axes[r, c].set_title(description, fontdict={'fontsize': fontsize}, loc='left')
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

        # set the rest of subplots to invisible
        mat_locs = [(r, c) for r, c in zip(rows, cols)]
        all_locs = [(r, c) for r in range(n_rows) for c in range(n_cols)]  
        nan_locs = set(all_locs) - set(mat_locs)

    else:
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
            im = axes[r, c].matshow(data)
            axes[r, c].set_title(description, fontdict={'fontsize': fontsize}, loc='left')
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

            # debug: colorbar under dev
            # todo: fix height mismatch
            mat_locs.append((r, c))
            if data.shape[0] >= data.shape[1]:
                plt.colorbar(im, cax=axes[r, c + 1])
                cbar_locs.append((r, c + 1))
            else:
                # plt.colorbar(im, cax=axes[r + 1, c], location='bottom')
                plt.colorbar(im, cax=axes[r + 1, c], orientation="horizontal")
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
