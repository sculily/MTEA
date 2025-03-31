import numpy as np
import pandas as pd
def csv():
    data = np.load('datasets/ad/34_smtp.npz', allow_pickle=True)
    X, y = data['X'], data['y']
    y = y.reshape(95156,1)
    z = np.hstack((X,y))
    pd.DataFrame(z).to_csv("datasets/ad/34_smtp.csv")
    #np.savetxt('datasets/ad/42_WBC.csv',z)

    print("z", z.shape)

if __name__ == '__main__':
    print('Start test...')
    csv()
    print('test completed')