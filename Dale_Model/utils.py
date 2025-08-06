
import numpy as np
from scipy.spatial.distance import cdist


def roll_idx(n):
    '''Create index that moves the first matrix row to the last'''
    idx = list(range(n))[1:n]
    idx.append(0)
    return idx 

def add_one(features=4, samples=5):
    # Check sample equation?
    XX = np.eye(features)
    XY = XX[:, roll_idx(features)]

    XX = np.tile(XX, (samples)).T
    XY = np.tile(XY, (samples)).T
    return XX, XY

def calc_dist(a, b):
    '''
    Requires:
    from scipy.spatial.distance import cdist'''
    return 1 - cdist(a, b, metric='cosine')

def H2I(H):
	""" Convert one-hot back to int """
	return np.where(H)[0][0]