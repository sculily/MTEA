import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

#iris = datasets.load_iris()
#R = np.array(iris.data)
'''
data = pd.read_csv('datasets/ad/glass_245(1).csv')
x_train = data.iloc[:, :3].to_numpy()
y_train = data['class']
current_class = 1
other_class = 0

'''

def plot_3D_PCA_Legend(new_point):
    data = pd.read_csv('datasets/ad/glass_245(1).csv')
    x_train = data.iloc[:, :3].to_numpy()
    y_train = data['class']
    x_train.astype(np.float32)
    # 进行PCA降维
    x_train = np.append(x_train, new_point, axis=0)
    y_train = np.append(y_train, [2.], axis=0)
    x_reduced = PCA(n_components=3).fit_transform(x_train)
    y = y_train

    #sp_names = np.unique(np.array(y)).tolist()
    #print("sp_names", sp_names)
    fig = plt.figure()
    ax = Axes3D(fig)
    # 3D散点图有标题，label，图例
    scatter = ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=y_train)
    ax.set_title('鸢尾花降维3维图')
    #ax.set_xlabel("PC1", size=18)
    #ax.set_ylabel("PC2", size=18)
    #ax.set_zlabel("PC3", size=18)
    # 添加图例名称到图标
    ax.legend(handles=scatter.legend_elements()[0],
              labels=['normal instance', 'outlier', 'new point'], loc="upper right")
    plt.savefig('3D_plot_1' + '.jpg')

def plot_3D_PCA_Legend_2():
    data = pd.read_csv('datasets/ad/Lymphography_012.csv')
    x_train = data.iloc[:, :3].to_numpy()
    y_train = data['label']
    x_train.astype(np.float32)
    current_class = 1
    other_class = 0
    # 进行PCA降维
    #x_train = np.append(x_train, new_point, axis=0)
    #y_train = np.append(y_train, [2.], axis=0)
    x_reduced = PCA(n_components=3).fit_transform(x_train)
    y = y_train

    #sp_names = np.unique(np.array(y)).tolist()
    #print("sp_names", sp_names)
    fig = plt.figure()
    ax = Axes3D(fig)
    # 3D散点图有标题，label，图例
    scatter = ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=y_train)
    ax.set_title('鸢尾花降维3维图')
    ax.set_xlabel("dimension 0", size=12)
    ax.set_ylabel("dimension 1", size=12)
    ax.set_zlabel("dimension 2", size=12)
    # 添加图例名称到图标
    #ax.legend(handles=scatter.legend_elements()[0],
              #labels=['normal instance', 'outlier', 'a query outlier'], loc="upper right")
    plt.savefig('3D_plot_Lym_012' + '.eps')

def plot_3D_PCA_Legend_3():
    data = pd.read_csv('datasets/ad/Lymphography_59.csv')
    x_train = data.iloc[:, :2].to_numpy()
    y_train = data['label']
    x_train.astype(np.float32)

    x = x_train
    y1 = y_train

    plt.scatter(x[:, 0], x[:, 1], c=y_train)
    plt.xlabel("dimension 5", size=12)
    plt.ylabel("dimension 9", size=12)
    
    plt.savefig('3D_plot_Lym_59' + '.eps')
    
    

if __name__ == '__main__':
    print('Start test...')
    plot_3D_PCA_Legend_3()
    print('test completed')  
