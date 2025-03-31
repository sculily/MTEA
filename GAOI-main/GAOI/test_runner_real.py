import os
import ast
import math
import logging
import pickle
from argparse import ArgumentParser
from time import time

import random
import numpy as np
import tensorflow as tf

random_seed = 266
random.seed(random_seed )  # set random seed for python
np.random.seed(random_seed )  # set random seed for numpy
tf.random.set_seed(random_seed )

import pandas as pd
from sklearn import metrics

from explainers.logger_utils import time_format
from tests.ADRealData import run_test, realdata 
from utils.argument_parser import DefaultReal


class Real(DefaultReal):
    def __call__(self, parser):
        parser = super().__call__(parser)
        return parser

def evaluation(exp_subspace_list, X, y):
    print("exp_subspace_list", exp_subspace_list)
    g_truth_df = pd.read_csv('datasets/ground_truth/glass_gt_sod.csv')

    ano_idx = np.where(y == 1)[0]

    precision_list = np.zeros(len(ano_idx))
    jaccard_list = np.zeros(len(ano_idx))
    recall_list = np.zeros(len(ano_idx))
    F1 = np.zeros(len(ano_idx))

    for ii, ano in enumerate(ano_idx):
        exp_subspace = list(exp_subspace_list[ii])
        gt_subspace_str = g_truth_df.loc[g_truth_df["ano_idx"] == ano]["exp_subspace"].values[0]
        gt_subspace = ast.literal_eval(gt_subspace_str)

        overlap = list(set(gt_subspace).intersection(set(exp_subspace)))
        union = list(set(gt_subspace).union(set(exp_subspace)))

        precision_list[ii] = len(overlap) / len(exp_subspace)
        jaccard_list[ii] = len(overlap) / len(union)
        recall_list[ii] = len(overlap) / len(gt_subspace)
        #F1[ii] = (2*precision_list[ii] * recall_list[ii])/(precision_list[ii] + recall_list[ii])
    #print("F1", (2*precision_list.mean() * recall_list.mean())/(precision_list.mean() + recall_list.mean()))

    F1 = (2*precision_list.mean() * recall_list.mean())/(precision_list.mean() + recall_list.mean())
    return precision_list.mean(), recall_list.mean(), jaccard_list.mean(), F1

def evaluation_od_auc(feature_weight, x, y):
    g_truth_df = pd.read_csv('datasets/ground_truth/glass_gt_sod.csv')

    ano_idx = np.where(y == 1)[0]
    dim = x.shape[1]

    auroc_list = np.zeros(len(ano_idx))
    aupr_list = np.zeros(len(ano_idx))
    for ii, ano in enumerate(ano_idx):
        score = feature_weight[ii]

        gt_subspace_str = g_truth_df.loc[g_truth_df["ano_idx"] == ano]["exp_subspace"].values[0]
        gt_subspace = ast.literal_eval(gt_subspace_str)
        gt = np.zeros(dim, dtype=int)
        gt[gt_subspace] = 1

        if len(gt_subspace) == dim:
            auroc_list[ii] = 1
            aupr_list[ii] = 1
        else:
            precision, recall, _ = metrics.precision_recall_curve(gt, score)
            aupr_list[ii] = metrics.auc(recall, precision)
            auroc_list[ii] = metrics.roc_auc_score(gt, score)

    return aupr_list.mean(), auroc_list.mean()

def weight2subspace(weight, ratio=0.7, num=-1):
    dim = len(weight)
    #threshold = np.sum(weight) / dim
    threshold = ratio * np.sum(weight)
    print("num", num)


    sorted_idx = np.argsort(weight)
    sorted_idx = [sorted_idx[dim - i - 1] for i in range(dim)]
    if num != -1:
        print("sorted_idx[:num]", sorted_idx[:num])
        exp_subspace = sorted_idx[:num]
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace
    
    print("sorted_idx", sorted_idx)

    tmp_s = 0
    exp_subspace = []
    for idx in sorted_idx:
        
        tmp_s += weight[idx]
        exp_subspace.append(idx)
        if tmp_s >= threshold:
            break
        
        #if weight[idx] >= threshold:
        #    exp_subspace.append(idx)
        print("exp_subspace", exp_subspace)
    exp_subspace = list(np.sort(exp_subspace))
    return exp_subspace


def get_exp_subspace(fea_weight_lst, real_exp_len=None):
    exp_subspace_lst = []
    n_ano = len(fea_weight_lst)
    dim = len(fea_weight_lst[0])
    r = math.sqrt(2 / dim)

    for ii in range(n_ano):
        fea_weight = fea_weight_lst[ii]
        #exp_subspace_lst.append(weight2subspace(fea_weight, num=real_exp_len[ii]))
        #exp_subspace_lst.append(weight2subspace(fea_weight, ratio = r))
        exp_subspace_lst.append(weight2subspace(fea_weight))
    return exp_subspace_lst

if __name__ == '__main__':
        conf = Real()
        conf = conf(ArgumentParser())
        args = conf.parse_args()
        gt_lst = []
        real_len_lst = []
        runs_metric_lst = []
        runs_metric_ls1 = []
        gt_str = pd.read_csv('datasets/ground_truth/Lymphography_gt_sod.csv')["exp_subspace"].values
        gt_lst.append([ast.literal_eval(gtt) for gtt in gt_str])
        for gt in gt_lst:
            real_len_lst.append([len(gtt) for gtt in gt])

        path = f"tests/logs/{vars(args)['dataset']}_{time_format(time())}"
        os.mkdir(path)

        with open(os.path.join(path, 'conf.txt'), 'w') as f:
            print('-------- PARAMETERS -------', file=f)
            for k in vars(args).keys():
                print('{} {}'.format(k, vars(args)[k]), file=f)
            print('---------------------------', file=f)

        print('Start test...')
        #run_test(path, **vars(args))
        fea_weight_lst, X, y = realdata(path, **vars(args))
        print("fea_weight_lst", fea_weight_lst)
        pickle.dump(fea_weight_lst, open(os.path.join(path, 'fea_weight_lst.joblib'), 'wb'))
        subspace_outputs = []
        real_len = real_len_lst[0]
        #subspace_outputs = get_exp_subspace(fea_weight_lst, real_exp_len=real_len)
        subspace_outputs = get_exp_subspace(fea_weight_lst)
        #print("subspace", subspace)
        #subspace_outputs.append(subspace)

        p, j, s, f1 = evaluation(subspace_outputs, X, y)
        aupr, auroc = evaluation_od_auc(fea_weight_lst, X, y)
        metric_lst = [p, j, s, f1]
        metric_lst1 = [aupr, auroc]
        runs_metric_lst.append(metric_lst)
        runs_metric_ls1.append(metric_lst1)
        logging.info(f'PRED: {runs_metric_lst}')

        print("**runs_metric_lst**", runs_metric_lst)
        print("**runs_metric_lst1**", runs_metric_ls1)

        print('test completed')