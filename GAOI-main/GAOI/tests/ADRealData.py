import logging
import pickle
from math import ceil
from time import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import scipy.io

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from explainers import AETabularMM
from utils.utils import getclass, getneighds, focal_loss, data_plot, cross_loss
from plot_3D import plot_3D_PCA_Legend, plot_3D_PCA_Legend_2, plot_3D_PCA_Legend_3
from models.ad_models import define_ad_model_complex, define_ad_attention # define_ad_model_complex

def realdata(path, **kwargs):
    epochs = kwargs.pop('epochs')
    epochs_exp = kwargs.pop('exp_epochs')
    batch_size = kwargs.pop('batch_size')
    batch_exp = kwargs.pop('exp_batch')
    n_adv = kwargs.pop('n_adv')
    n_dim = kwargs.pop('dim_number')
    loss_weights = kwargs.pop('loss_weights')
    ns = kwargs.pop('n_same')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    dataset = kwargs.pop('dataset')
    n_class = kwargs.pop('n_class')

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')

    if dataset == '1_glass':
        data = pd.read_csv('datasets/ad/1_glass.csv')
        columns = data.columns
        x_train = data.iloc[:, :7].to_numpy()
        y_train = data['class']
        current_class = 1
        other_class = 0
        c_columns = None
    elif dataset == '2_Wilt':
        data = pd.read_csv('datasets/ad/2_Wilt.csv')
        columns = data.columns
        x_train = data.iloc[:, :5].to_numpy()
        y_train = data['class']
        current_class = 1
        other_class = 0
        c_columns = None
    elif dataset == '3_Lymphography':
        data = pd.read_csv('datasets/ad/3_Lymphography.csv')
        columns = data.columns
        x_train = data.iloc[:, :10].to_numpy()
        y_train = data['label']
        current_class = 1
        other_class = 0
        c_columns = None
    elif dataset == '4_thyroid':
        data = pd.read_csv('datasets/ad/4_thyroid.csv')
        columns = data.columns
        x_train = data.iloc[:, :7].to_numpy()
        y_train = data['class']
        current_class = 1
        other_class = 0
        c_columns = None
    elif dataset == '5_yeast':
        data = pd.read_csv('datasets/ad/5_yeast.csv')
        columns = data.columns
        x_train = data.iloc[:, :8].to_numpy()
        y_train = data['class']
        current_class = 1
        other_class = 0
        c_columns = None
    else:
        raise NotImplementedError(f'Dataset {dataset} not yet implemented')

    ano_idx = np.where(y_train == 1)[0]
    iterator = range(len(ano_idx))
    fea_weight_lst = []
    for ii in iterator:
        idx = ano_idx[ii]
        fea_weight = run_test(path, x_train, y_train, current_class, other_class, ns, no, epochs, epochs_exp, batch_size, batch_exp, n_adv, n_dim, loss_weights, lr, threshold, n_class, columns, c_columns)
        fea_weight_lst.append(fea_weight)
    logging.info(f'PRED: {fea_weight_lst}')
    return fea_weight_lst, x_train, y_train



def run_test(path, x_train, y_train, current_class, other_class, ns, no, epochs, epochs_exp, batch_size, batch_exp, n_adv, n_dim, loss_weights, lr, threshold, n_class, columns, c_columns):

    ad_model_opt = tf.keras.optimizers.Adam()
    exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)
    explanations = []
    masks = []

    # -------------------- Try with more samples ---------------------------
    
    train_s, train_l, classes, invclasses = getclass(x_train, y_train.astype(np.int32), current_class,
                                                     other_class)

    img_id = np.where(train_l == 0)[0][0]
    print("img_id", img_id)

    test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no, columns=c_columns)
    test_labels_expl = classes[test_labels_expl]
    
    test_images_expl = test_images_expl.astype(np.float32)


    x_train = x_train.astype(np.float32)
    sample_to_explain = train_s[img_id: img_id + 1]
    print("sample_to_explain", sample_to_explain)
    # ----------------------------------------------------------------------

    x_train_ext = test_images_expl.copy()

    y_train_ext = test_labels_expl.copy()

    in_shape = x_train.shape[1]

    # Define the ad model
    ad_model = define_ad_model_complex(x_train[0].shape)

    ad_model.compile(optimizer=ad_model_opt, loss=focal_loss)
    explainer = AETabularMM.TabularMM(ad_model, in_shape, optimizer=exp_opt)




    for i in range(2):
        print('--------------------- ADV EPOCH: {} -------------------'.format(i))
        start_time = time()
        ad_model.trainable = True
        
        x_train_ext[0] = sample_to_explain
        train_s, train_l, classes, invclasses = getclass(x_train_ext, y_train_ext.astype(np.int32), current_class, other_class)
        img_id = np.where(train_l == 0)[0][0]
        test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no, columns=c_columns)
        test_labels_expl = classes[test_labels_expl]
        test_images_expl = test_images_expl.astype(np.float32)
        x_train_ext = test_images_expl.copy()
        y_train_ext = test_labels_expl.copy()

        ad_model.fit(x_train_ext, y_train_ext, epochs=epochs, batch_size=batch_size, verbose=1)
        ad_model.trainable = False
        ad_model.evaluate(x_train_ext, y_train_ext)


        #Early-stopping
        pred = ad_model.predict(sample_to_explain)[:, 0]
        if pred < 0.6:
            logging.info(f'PRED: {pred}')
            logging.info(f'EARLY STOPPING EPOCH {i}')
            break

        explainer.explain(test_images_expl, test_labels_expl, batch_size=batch_exp,
                          epochs=epochs_exp, loss_weights=loss_weights)  # loss_weights=[1., 0.2, .4]
        tot_time = time() - start_time

        print('Elapsed time: ', tot_time)
        logging.info(f'Elapsed time explanation {i}: {tot_time}')

        explanations.append(explainer)
        new_sample = explainer.PATCH(sample_to_explain.reshape(1, -1))
        new_sample = new_sample.numpy()
        mask, choose = explainer.MASKGEN(sample_to_explain.reshape(1, -1))
        masks.append(mask.numpy())
        x_train_ext = np.append(x_train_ext, new_sample, axis=0)
        y_train_ext = np.append(y_train_ext, [1.], axis=0)
        sample_to_explain = new_sample

    new_sample, dims = explainer.return_explanation(in_shape, sample_to_explain.reshape(1, -1), threshold=threshold)
    #featue_subspace = []
    #featue_subspace.append(dims)
    #print("***dims",np.array(dims))

    #plot_3D_PCA_Legend(new_point=new_sample)
    #data_plot(x_train, y_train, new_point=new_sample[0], dimensions=[1, 2, 6, 7, 9], name=os.path.join(path, f'explanation_abod'), train=True, features_name=columns)
    #plt.close('all')
    return np.array(dims)[0]