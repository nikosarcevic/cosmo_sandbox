import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.patches import Ellipse
import cmasher as cmr

def plot_shapes(x, y, e1, e2, ax, style='arrow', scale=None, color='yellowgreen', stretch=3.0):
    """
    Plot galaxy shapes (ellipticities) as arrows or ellipses.

    Parameters
    ----------
    x, y : ndarray
        Coordinates of galaxy positions.
    e1, e2 : ndarray
        Ellipticity components (e1, e2).
    ax : matplotlib.axes.Axes
        Axes to draw on.
    style : {'arrow', 'ellipse'}
        'arrow' uses quiver, 'ellipse' uses Ellipse patches.
    scale : float or None
        Shape size scaling. If None, estimated from data spread.
    color : str
        Color of shapes.
    stretch : float
        Ellipse elongation exaggeration factor (applied to major axis).
    """
    ax.set_aspect('equal')

    if scale is None:
        xspread = np.ptp(x)
        yspread = np.ptp(y)
        scale = 0.03 * max(xspread, yspread)  # auto-scaling

    if style == 'arrow':
        ax.quiver(x, y, e1, e2, angles='xy', scale_units='xy',
                  scale=1/scale, color=color, width=0.003)

    elif style == 'ellipse':
        for xi, yi, ei1, ei2 in zip(x, y, e1, e2):
            e = np.hypot(ei1, ei2)
            if e > 1:
                e = np.hypot(ei1, ei2)
                e = min(e, 0.9)  # clip large ellipticity to avoid degenerate ellipses

            angle = 0.5 * np.arctan2(ei2, ei1) * 180 / np.pi  # degrees
            major = scale * (1 + e) * stretch
            minor = scale * (1 - e)

            ell = Ellipse(
                xy=(xi, yi), width=major, height=minor,
                angle=angle, edgecolor=color, facecolor='none', linewidth=0.6
            )
            ax.add_patch(ell)
    else:
        raise ValueError("style must be 'arrow' or 'ellipse'")
