import numpy as np 

from matplotlib import pyplot as plt
plt.ioff()

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import matplotlib
import matplotlib.style as mplstyle
mplstyle.use('fast')
matplotlib.use('agg')

# FIXME: slow function!
def _img_to_array(im):
    # NOTE: https://stackoverflow.com/questions/31393769/getting-an-rgba-array-from-a-matplotlib-image
    # NOTE: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    fig = im.get_figure() # CAUTION: not documented

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()

    # convert to a NumPy array, ignore A channel
    X = np.asarray(buf)

    return X[:,:,(0,1,2)]

def _remove_head_cartoon(fig_arr):
    mask = _get_head_mask()
    masked_img = fig_arr.copy()
    masked_img[~mask] = 0 # = 0 gives black outside
    return masked_img

# mask mne head cartoon with a circular mask
# CAUTION: FIXME: these numbers are empirical! plot your images and check!
def _get_head_mask(w=100, h=100):
    center = (int(w/2), int(h/2))
    mask = _create_circular_mask(h, w, center=center, radius=33)
    return mask

# https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def _create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask