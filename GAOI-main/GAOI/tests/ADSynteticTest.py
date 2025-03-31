import logging
import pickle
from math import ceil
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
sys.path.append("/home/lab/miniconda3/envs/tf2/MMOAM-main/MaskingModelExplainer")
from explainers import AETabularMM
from utils.utils import getclass, getneighds, focal_loss, data_plot
from models.ad_models import define_ad_model


def run_test(path, **kwargs):
    epochs = kwargs.pop('epochs')
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    batch_size = kwargs.pop('batch_size')
    n_norm = kwargs.pop('n_samples_num')
    n_anorm = kwargs.pop('a_samples_num')
    n_adv = kwargs.pop('n_adv')
    n_dim = kwargs.pop('dim_number')
    # loss_weights = [1., 0.4, .5]
    loss_weights = kwargs.pop('loss_weights')
    n_mean = kwargs.pop('n_mean')
    n_std = kwargs.pop('n_std')
    dist_std = kwargs.pop('dist_std')
    ns = kwargs.pop('n_same') # 0
    no = kwargs.pop('n_other') # 1
    lr = kwargs.pop('learning_rate')
    anom_dims = kwargs.pop('anom_dims')
    threshold = kwargs.pop('threshold')

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), format='%(message)s', level=logging.INFO)

    # Generate a syntetic dataset with mean 3 and variance 2
    x_train = np.random.normal(n_mean, n_std, (n_norm + n_anorm, n_dim))
    dims = np.random.permutation(np.arange(n_dim))

    # Create an anomalous sample
    logging.info('\nPoint before anormality introduction: \n' + str(x_train[-1]))
    mod_dir = np.random.randint(0, 2, (anom_dims)) * 2 - 1
    logging.info(f'Mod dir dim {dims[:anom_dims]}: {mod_dir}\n')
    x_train[n_norm:, dims[:anom_dims]] += mod_dir * dist_std * n_std
    real_dims = []
    real_dims.append(dims[:anom_dims])
    print("dims[:anom_dims]", dims[:anom_dims])
    print("real_dims", real_dims)
    x_train = x_train.astype(np.float32)

    logging.info('\nPoint after anormality introduction: \n' + str(x_train[-1]))

    y_train = np.zeros(x_train.shape[0], dtype=np.int32)
    y_train[n_norm:] = 1
    y_train = y_train.astype(np.float32)

    #print("x_train", x_train) #[[]] (101,10)
    #print("y_train", y_train) # [] (101,)

    data_plot(x_train, y_train, name=os.path.join(path, f'x_train_y_train'))
    y_train = y_train.reshape(1010,1)
    z = np.hstack((x_train,y_train))
    pd.DataFrame(z).to_csv("datasets/ad/synthetic_1024_dim.csv")

    in_shape = x_train.shape[1]

    # Define the ad model
    #print("x_train[0]", x_train[0])
    ad_model = define_ad_model(x_train[0].shape)

    ad_model_opt = tf.keras.optimizers.Adam()
    exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)

    explanations = []
    masks = []

    x_train_ext = x_train.copy()
    y_train_ext = y_train.copy()

    # -------------------- Try with more samples ---------------------------
    current_class = 1
    other_class = 0
    train_s, train_l, classes, invclasses = getclass(x_train, y_train.astype(np.int32), current_class,
                                                     other_class)
    data_plot(train_s, train_l, name=os.path.join(path, f'train_s_train_l'))
    img_id = np.where(train_l == 0)[0][0]
    #print("np.where(train_l == 0)",np.where(train_l == 0)) 输出： (array([100]),)
    #print("img_id",img_id) 输出： 100

    test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no)
    #print("test_labels_expl_1", test_labels_expl) 0,1
    test_labels_expl = classes[test_labels_expl]
    #print("test_labels_expl_2", test_labels_expl) 1,0
    test_images_expl = test_images_expl.astype(np.float32)
    #print("test_images_expl_2", test_images_expl)
    data_plot(test_images_expl, test_labels_expl, name=os.path.join(path, f'test_images_expl_test_labels_expl'))
    #ad_model.compile(optimizer=ad_model_opt, loss=focal_loss, metrics=['accuracy'])

    explainer = AETabularMM.TabularMM(ad_model, in_shape, optimizer=exp_opt)
    #sample_to_explain = x_train[np.where(y_train == 1)]
    # ----------------------------------------------------------------------

    for i in range(n_adv):
        print('--------------------- ADV EPOCH: {} -------------------'.format(i))
        sample_to_explain = x_train[np.where(y_train == 1)] #outlier [[]]
        #print("sample_to_explain",sample_to_explain)

        start_time = time()
        ad_model.trainable = True
        ad_model.compile(optimizer=ad_model_opt, loss=focal_loss, metrics=['accuracy'])
        ad_model.fit(x_train_ext, y_train_ext, batch_size=batch_size, epochs=epochs)
        ad_model.trainable = False
        # Early-stopping
        pred = ad_model.predict(sample_to_explain)[:, 0] # the adversarial net
        if pred < 0.5:
            logging.info(f'PRED: {pred}')
            logging.info(f'EARLY STOPPING EPOCH {i}')
            break

        explainer.explain(test_images_expl, test_labels_expl, batch_size=batch_exp,
                          epochs=epochs_exp, loss_weights=loss_weights)
        tot_time = time() - start_time
        print('Elapsed time: ', tot_time)
        logging.info(f'Elapsed time explanation {i}: {tot_time}')
        explanations.append(explainer)
        new_sample = explainer.PATCH(sample_to_explain.reshape(1, -1))
        #print("new_sample", new_sample)
        new_sample = new_sample.numpy()
        print("new_sample_{i}", new_sample)
        logging.info('Adversarial point {}: '.format(i) + str(new_sample))
        logging.info('Adversarial classification {}: '.format(i) + str(ad_model.predict(new_sample)))
        mask, choose = explainer.MASKGEN(sample_to_explain.reshape(1, -1))
        #print("mask",mask)
        #print("choose",choose)
        masks.append(mask.numpy())
        #print("masks",masks)
        x_train_ext = np.append(x_train_ext, new_sample, axis=0)
        y_train_ext = np.append(y_train_ext, [1.], axis=0)
        print("y_train_ext", y_train_ext)

        data_plot(x_train, y_train, new_point=new_sample[0], name=os.path.join(path, f'adv_point_{i}'), train=True)
        data_plot(test_images_expl, test_labels_expl, new_point=new_sample[0], name=os.path.join(path, f'neighbourhood_{i}'), train=True)

        plt.figure(2, figsize=(4 * min(n_adv, 5), 3.8 * ceil(n_adv / 5)))
        plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        plt.bar(np.arange(0, n_dim, dtype=np.int32), mask[0])
        plt.title('Try n.: {}'.format(i + 1))
        plt.tight_layout()
        plt.figure(3, figsize=(4 * min(n_adv, 5), 3.8 * ceil(n_adv / 5)))
        plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        plt.bar(np.arange(0, n_dim, dtype=np.int32), choose[0])
        plt.title('Try n.: {}'.format(i + 1))
        plt.tight_layout()



    pickle.dump(x_train_ext, open(os.path.join(path, 'x_train_ext.joblib'), 'wb'))
    pickle.dump(y_train_ext, open(os.path.join(path, 'y_train_ext.joblib'), 'wb'))
    pickle.dump(test_images_expl, open(os.path.join(path, 'chosen_points_x.joblib'), 'wb'))
    pickle.dump(test_labels_expl, open(os.path.join(path, 'chosen_points_y.joblib'), 'wb'))
    pickle.dump(masks, open(os.path.join(path, 'masks.joblib'), 'wb'))
    plt.figure(2)
    plt.savefig(os.path.join(path, 'explanations.eps'))
    plt.savefig(os.path.join(path, 'explanations.jpg'))
    plt.figure(3)
    plt.savefig(os.path.join(path, 'choose.eps'))
    plt.savefig(os.path.join(path, 'choose.jpg'))
    new_sample, dims = explainer.return_explanation(sample_to_explain.reshape(1, -1), threshold=threshold)
    print("new_sample", new_sample)
    print("dims", dims)


    #data_plot(x_train, y_train, new_point=new_sample[0], dimensions=np.where(dims > threshold)[1],
    #          name=os.path.join(path, f'explanation'), train=True)
    pickle.dump(np.argwhere(dims>=threshold), open('choose.pickle', 'wb'))
    plt.close('all')