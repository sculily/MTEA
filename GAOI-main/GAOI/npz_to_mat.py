import numpy as np
import pandas as pd
import scipy.io as io
def mat():
    io.savemat('datasets/trans/17_InternetAds.mat', mdict=np.load('datasets/trans/17_InternetAds.npz', allow_pickle=True))

if __name__ == '__main__':
    print('Start test...')
    mat()
    print('test completed')