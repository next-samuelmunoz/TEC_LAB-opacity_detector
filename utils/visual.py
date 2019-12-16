# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt
import numpy as np
from shapely import geometry


def imshow(img, title=None, roi=None, points=None, weights=None):
    """Plot OpenCV BRG images with matplotlib
    By default, OpenCV uses an BGR format (common one, RGB)

    points: [(x,y)]
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    if roi:
        roi = geometry.Polygon(roi)
        x,y = roi.exterior.xy
        plt.plot(x, y, color='#00ffff', alpha=1.0, linewidth=3, solid_capstyle='round', zorder=2)
    if points:
        if weights:
            weights, points = zip(*sorted(zip(weights,points)))  # Reverse order so important points overlap others
            s = ((np.array(weights)*20)**2)
            c = weights
        else:
            s = None
            c = 'red'
        x,y = zip(*points)
        plt.scatter(x, y, s=s, cmap='hot', c=c, marker='o', alpha=0.7 )
        if weights:
            plt.colorbar()
    plt.show()
